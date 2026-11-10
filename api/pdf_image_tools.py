"""Place an uploaded raster on a new, independent PDF optional-content layer.

The original PDF is copied on disk and updated incrementally. Only the uploaded
image is decoded; existing pages, raster images and vector objects are retained.
"""
from __future__ import annotations

import math
import os
import re
import tempfile
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from .pdf_engine import MILLIMETRES_PER_INCH, POINTS_PER_INCH, _check_cancelled, _pymupdf


class PdfImageToolError(ValueError):
    """Invalid PDF, uploaded image or requested placement (HTTP 422)."""


_PILLOW_IMAGE_LOCK = threading.Lock()
_COPY_CHUNK_BYTES = 8 * 1024 * 1024
_IMAGE_OPERATOR = re.compile(rb"/([^\s/]+)\s+Do\b")
DEFAULT_MAX_IMAGE_PIXELS = 268_435_456


def _number(value: Any, name: str, *, positive: bool) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise PdfImageToolError(f"{name} 必须是有限数值") from exc
    if isinstance(value, bool) or not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise PdfImageToolError(f"{name} 必须是{'大于 0' if positive else '不小于 0'}的有限数值")
    return result


def _check_image_header(image_path: Path, max_image_pixels: int) -> None:
    from PIL import Image

    try:
        with _PILLOW_IMAGE_LOCK:
            previous_limit = Image.MAX_IMAGE_PIXELS
            Image.MAX_IMAGE_PIXELS = None
            try:
                with Image.open(image_path) as header:
                    width, height = header.size
                    # Reading size does not call load() or allocate the decoded
                    # raster. Apply our explicit upload limit before Pixmap.
                    if width <= 0 or height <= 0:
                        raise PdfImageToolError("图片像素尺寸必须大于 0")
                    if width * height > max_image_pixels:
                        raise PdfImageToolError(
                            f"上传图片像素数 {width}×{height}={width * height:,} 超过上限 {max_image_pixels:,}；"
                            "请减小图片像素尺寸，或调整 CMYK_API_MAX_IMAGE_PIXELS"
                        )
            finally:
                Image.MAX_IMAGE_PIXELS = previous_limit
    except PdfImageToolError:
        raise
    except Exception as exc:
        raise PdfImageToolError("无法读取图片文件头；请使用有效的 PNG、JPEG、TIFF、BMP 或 WebP 图片") from exc


@contextmanager
def _image_input(image_path: Path, temporary_directory: Path, max_image_pixels: int):
    """Prefer MuPDF's native PNG/JPEG/TIFF/BMP decoders; WebP uses Pillow.

    Pillow is only a fallback for this uploaded raster, never for the PDF.
    Its pixel threshold is disabled briefly while loading the fallback image,
    and restored afterwards. Large PDF pages do not pass through Pillow.
    """
    _check_image_header(image_path, max_image_pixels)
    pymupdf = _pymupdf()
    try:
        image = pymupdf.Pixmap(image_path)
    except Exception:
        try:
            from PIL import Image

            converted_path = temporary_directory / "uploaded-image.png"
            with _PILLOW_IMAGE_LOCK:
                previous_limit = Image.MAX_IMAGE_PIXELS
                Image.MAX_IMAGE_PIXELS = None
                try:
                    with Image.open(image_path) as original:
                        original.seek(0)
                        original.load()
                        # PNG cannot store CMYK; retain CMYK through a TIFF
                        # fallback rather than introducing an RGB conversion.
                        if original.mode == "CMYK":
                            converted_path = temporary_directory / "uploaded-image.tif"
                            original.save(converted_path, format="TIFF", compression="tiff_deflate")
                        else:
                            has_alpha = "A" in original.getbands() or "transparency" in original.info
                            with original.convert("RGBA" if has_alpha else "RGB") as converted:
                                converted.save(converted_path, format="PNG")
                finally:
                    Image.MAX_IMAGE_PIXELS = previous_limit
            image = pymupdf.Pixmap(converted_path)
        except Exception as exc:
            raise PdfImageToolError("图片损坏或格式不支持；请使用 PNG、JPEG、TIFF、BMP 或 WebP 图片") from exc
        chosen_path = converted_path
    else:
        chosen_path = image_path
    width, height, alpha = int(image.width), int(image.height), bool(image.alpha)
    del image
    if width <= 0 or height <= 0:
        raise PdfImageToolError("图片像素尺寸必须大于 0")
    yield chosen_path, width, height, alpha


def _display_image_matrix(page, displayed_rect):
    """Map image-unit coordinates to raw PDF coordinates.

    A temporary zero rotation exposes MuPDF's full CropBox/UserUnit transform.
    We then apply the inverse display rotation explicitly. Replacing only the
    newly generated image stream avoids rotated-page insertion quirks in
    PyMuPDF, especially when UserUnit is not one.
    """
    pymupdf = _pymupdf()
    rotation = page.rotation
    display_width, display_height = page.rect.width, page.rect.height
    width, height = (display_height, display_width) if rotation in (90, 270) else (display_width, display_height)
    inverse_rotation = {
        0: pymupdf.Matrix(1, 0, 0, 1, 0, 0),
        90: pymupdf.Matrix(0, -1, 1, 0, 0, height),
        180: pymupdf.Matrix(-1, 0, 0, -1, width, height),
        270: pymupdf.Matrix(0, 1, -1, 0, width, 0),
    }[rotation]
    try:
        if rotation:
            page.set_rotation(0)
        to_pdf = inverse_rotation * ~page.transformation_matrix
    finally:
        if rotation:
            page.set_rotation(rotation)
    lower_left = displayed_rect.bl * to_pdf
    lower_right = displayed_rect.br * to_pdf
    upper_left = displayed_rect.tl * to_pdf
    return (
        lower_right.x - lower_left.x, lower_right.y - lower_left.y,
        upper_left.x - lower_left.x, upper_left.y - lower_left.y,
        lower_left.x, lower_left.y,
    )


def _replace_image_reference(document, page, name: str, image_xref: int) -> None:
    resource_type, resources = document.xref_get_key(page.xref, "Resources")
    owner = int(resources.split()[0]) if resource_type == "xref" else page.xref
    prefix = "" if resource_type == "xref" else "Resources/"
    object_type, objects = document.xref_get_key(owner, prefix + "XObject")
    if object_type == "xref":
        document.xref_set_key(int(objects.split()[0]), name, f"{image_xref} 0 R")
    else:
        document.xref_set_key(owner, prefix + "XObject/" + name, f"{image_xref} 0 R")


def _isolate_original_graphics_state(document, page) -> None:
    # Existing content can leave a root-level transform, clip or opacity in
    # force even when its q/Q count is balanced. Enclose it as a whole so the
    # added image starts in the page's default graphics state.
    page.wrap_contents()
    contents = page.get_contents()
    prefix, suffix = document.get_new_xref(), document.get_new_xref()
    document.update_object(prefix, "<<>>")
    document.update_object(suffix, "<<>>")
    document.update_stream(prefix, b"q\n")
    document.update_stream(suffix, b"\nQ\n")
    document.xref_set_key(page.xref, "Contents", "[" + " ".join(f"{xref} 0 R" for xref in [prefix, *contents, suffix]) + "]")


def place_image_on_pdf(
    source_path: str | Path,
    image_path: str | Path,
    output_path: str | Path,
    *,
    page_number: int = 1,
    x_mm: float,
    y_mm: float,
    width_mm: float,
    height_mm: float,
    layer_name: str,
    cancelled: Callable[[], bool] | None = None,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> dict[str, Any]:
    """Insert a raster exactly into a displayed-CropBox rectangle in mm.

    ``page_number`` is one-based. The image is stretched to the requested width
    and height, remains upright in the displayed page and retains its intrinsic
    pixel dimensions and alpha. Every call creates a fresh OCG, even if another
    group already has the requested name. Multi-frame images use frame one.
    """
    source, image_path, output = Path(source_path), Path(image_path), Path(output_path)
    if not source.is_file():
        raise PdfImageToolError("输入 PDF 文件不存在")
    if not image_path.is_file():
        raise PdfImageToolError("上传图片不存在")
    if output.resolve() in {source.resolve(), image_path.resolve()}:
        raise PdfImageToolError("输出路径不能覆盖输入文件")
    if output.exists():
        raise FileExistsError(output)
    if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
        raise PdfImageToolError("page 必须是从 1 开始的整数")
    if isinstance(max_image_pixels, bool) or not isinstance(max_image_pixels, int) or max_image_pixels < 1:
        raise PdfImageToolError("max_image_pixels 必须是大于 0 的整数")
    x = _number(x_mm, "x_mm", positive=False)
    y = _number(y_mm, "y_mm", positive=False)
    width = _number(width_mm, "width_mm", positive=True)
    height = _number(height_mm, "height_mm", positive=True)
    if not isinstance(layer_name, str) or not layer_name.strip():
        raise PdfImageToolError("layer_name 不能为空")
    name = layer_name.strip()
    if "\x00" in name:
        raise PdfImageToolError("layer_name 不能包含空字符")
    _check_cancelled(cancelled)
    pymupdf = _pymupdf()
    try:
        with pymupdf.open(source) as original:
            if not original.is_pdf:
                raise PdfImageToolError("输入文件不是 PDF")
            if original.needs_pass or original.xref_get_key(-1, "Encrypt")[0] not in {"null", "unknown"}:
                raise PdfImageToolError("暂不支持加密 PDF")
            if page_number > original.page_count:
                raise PdfImageToolError(f"page 超出范围：PDF 共 {original.page_count} 页")
            page_count = original.page_count
            page = original[page_number - 1]
            page_width_pt, page_height_pt = float(page.rect.width), float(page.rect.height)
            scale = POINTS_PER_INCH / MILLIMETRES_PER_INCH
            rect = pymupdf.Rect(x * scale, y * scale, (x + width) * scale, (y + height) * scale)
            if rect.x1 > page_width_pt + 0.0001 or rect.y1 > page_height_pt + 0.0001:
                raise PdfImageToolError("图片矩形必须完整位于所选页面的可见 CropBox 内，不能超出页面")
            if not all(math.isfinite(value) for value in (page_width_pt, page_height_pt)) or min(page_width_pt, page_height_pt) <= 0:
                raise PdfImageToolError("PDF 页面尺寸无效")
    except PdfImageToolError:
        raise
    except Exception as exc:
        raise PdfImageToolError(f"无法读取 PDF：{exc}") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pdf-image-", dir=output.parent) as temporary:
        directory = Path(temporary)
        with _image_input(image_path, directory, max_image_pixels) as (chosen_image, pixel_width, pixel_height, has_alpha):
            _check_cancelled(cancelled)
            working = directory / "working.pdf"
            with source.open("rb") as reader, working.open("wb") as writer:
                while chunk := reader.read(_COPY_CHUNK_BYTES):
                    _check_cancelled(cancelled)
                    writer.write(chunk)
            try:
                with pymupdf.open(working) as document:
                    page = document[page_number - 1]
                    matrix = _display_image_matrix(page, rect)
                    _isolate_original_graphics_state(document, page)
                    new_layer = document.add_ocg(name, on=True, usage="Artwork")
                    document.xref_set_key(new_layer, "CMYKRecipeOCGId", "(" + uuid.uuid4().hex + ")")
                    before_image = document.xref_length()
                    image_xref = page.insert_image(rect, filename=chosen_image, keep_proportion=False, overlay=True, oc=new_layer)
                    content_xref = page.get_contents()[-1]
                    content = document.xref_stream(content_xref)
                    operator = _IMAGE_OPERATOR.search(content)
                    if operator is None:
                        raise PdfImageToolError("无法定位新插入的图片对象")
                    image_name = operator[1].decode("ascii")
                    if image_xref < before_image:
                        # Reusing an existing image must not attach the old
                        # image to a different OCG or alter its old visibility.
                        clone = document.get_new_xref()
                        document.update_object(clone, "<<>>")
                        document.xref_copy(image_xref, clone)
                        document.xref_set_key(clone, "OC", f"{new_layer} 0 R")
                        _replace_image_reference(document, page, image_name, clone)
                        image_xref = clone
                    matrix_text = " ".join(format(value, ".12g") for value in matrix)
                    document.update_stream(content_xref, f"q\n{matrix_text} cm\n/{image_name} Do\nQ\n".encode("ascii"), compress=True)
                    _check_cancelled(cancelled)
                    if document.can_save_incrementally():
                        document.saveIncr()
                        finished = working
                    else:
                        finished = directory / "finished.pdf"
                        document.save(finished, garbage=0, clean=False, deflate=False)
                _check_cancelled(cancelled)
                try:
                    os.replace(finished, output)
                except PermissionError:
                    # Some Windows file providers retain a read handle after
                    # MuPDF closes an incremental document. A bounded disk copy
                    # also works there; do not turn it into a whole-file bytes.
                    created = False
                    try:
                        with finished.open("rb") as reader, output.open("xb") as writer:
                            created = True
                            while chunk := reader.read(_COPY_CHUNK_BYTES):
                                _check_cancelled(cancelled)
                                writer.write(chunk)
                    except BaseException:
                        if created:
                            output.unlink(missing_ok=True)
                        raise
            except PdfImageToolError:
                raise
            except Exception as exc:
                from .pdf_engine import PdfProcessingCancelled

                if isinstance(exc, PdfProcessingCancelled):
                    raise
                raise PdfImageToolError(f"无法插入图片或写入 PDF：{exc}") from exc
    return {
        "kind": "pdf", "path": str(output), "byteLength": output.stat().st_size,
        "pageCount": page_count, "pageNumber": page_number,
        "newLayerId": str(new_layer), "name": name, "imageXref": image_xref,
        "placementMm": {"x": x, "y": y, "width": width, "height": height},
        "pageSizeMm": {"width": page_width_pt / scale, "height": page_height_pt / scale},
        "imagePixels": {"width": pixel_width, "height": pixel_height, "hasAlpha": has_alpha},
    }
