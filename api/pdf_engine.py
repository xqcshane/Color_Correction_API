from __future__ import annotations

import math
import os
import re
import tempfile
from io import BytesIO
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


POINTS_PER_INCH = 72.0
MILLIMETRES_PER_INCH = 25.4
DEFAULT_TILE_SIZE = 2048
DEFAULT_BLEED = 2

_ARTWORK_WORDS = re.compile(
    r"(?:print[\s_-]*artwork|artwork|texture|pattern|design|印花|纹理|图案|花型|面料)",
    re.IGNORECASE,
)
_TECHNICAL_WORDS = re.compile(
    r"(?:cut[\s_-]*line|cutline|notch|seam|marker|technical|刀线|裁片|裁剪|剪口|辅助线)",
    re.IGNORECASE,
)


class PdfEngineError(RuntimeError):
    pass


class PdfEngineUnavailable(PdfEngineError):
    pass


class PdfProcessingCancelled(PdfEngineError):
    pass


def _pymupdf():
    try:
        import pymupdf  # type: ignore
    except ImportError:
        try:
            import fitz as pymupdf  # type: ignore
        except ImportError as exc:
            raise PdfEngineUnavailable(
                "PDF 服务需要 PyMuPDF；请先安装 api/requirements.txt。"
            ) from exc
    return pymupdf


def _numpy():
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise PdfEngineUnavailable(
            "PDF 分块服务需要 NumPy；请先安装 api/requirements.txt。"
        ) from exc
    return np


def _check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled and cancelled():
        raise PdfProcessingCancelled("处理已取消")


@dataclass(frozen=True)
class PdfLayer:
    xref: int
    name: str
    default_visible: bool
    intent: tuple[str, ...] = ()
    usage: str = ""
    order: int = 0
    detected_artwork: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.xref),
            "xref": self.xref,
            "name": self.name,
            "defaultVisible": self.default_visible,
            "intent": list(self.intent),
            "usage": self.usage,
            "order": self.order,
            "detectedArtwork": self.detected_artwork,
        }


@dataclass(frozen=True)
class PageGeometry:
    index: int
    width_pt: float
    height_pt: float
    width_mm: float
    height_mm: float
    rotation: int
    mediabox: tuple[float, float, float, float]
    cropbox: tuple[float, float, float, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "widthPt": self.width_pt,
            "heightPt": self.height_pt,
            "widthMm": self.width_mm,
            "heightMm": self.height_mm,
            "rotation": self.rotation,
            "mediabox": list(self.mediabox),
            "cropbox": list(self.cropbox),
        }


@dataclass(frozen=True)
class TilePlan:
    index: int
    total: int
    page_index: int
    page_width_px: int
    page_height_px: int
    x: int
    y: int
    width: int
    height: int
    outer_x: int
    outer_y: int
    outer_width: int
    outer_height: int
    dpi: float

    @property
    def core_in_outer(self) -> tuple[int, int, int, int]:
        return (
            self.x - self.outer_x,
            self.y - self.outer_y,
            self.width,
            self.height,
        )


@dataclass
class TileContext:
    page_index: int
    layer_xref: int
    layer_name: str
    tile: TilePlan
    stats_region: tuple[int, int, int, int]


@dataclass
class TileResult:
    cmyk: bytes | bytearray | memoryview
    preview_rgba: bytes | bytearray | memoryview | None = None
    stats: Mapping[str, Any] = field(default_factory=dict)


class TileMapper(Protocol):
    def __call__(
        self,
        rgba: bytes,
        width: int,
        height: int,
        context: TileContext,
    ) -> TileResult | Mapping[str, Any]: ...


@dataclass(frozen=True)
class LayerImage:
    xref: int
    smask: int
    width: int
    height: int
    page_index: int
    layer_xref: int
    layer_name: str
    oc_xref: int


def choose_artwork_layer(layers: Sequence[PdfLayer]) -> int | None:
    if not layers:
        return None

    def score(layer: PdfLayer) -> tuple[int, int]:
        name = layer.name.strip()
        value = 0
        if _ARTWORK_WORDS.search(name):
            value += 100
        if _TECHNICAL_WORDS.search(name):
            value -= 120
        if layer.usage.lower() == "artwork":
            value += 10
        if layer.default_visible:
            value += 2
        return value, -layer.order

    best = max(layers, key=score)
    return best.xref if score(best)[0] > 0 else layers[0].xref


def _rect_tuple(rect: Any) -> tuple[float, float, float, float]:
    return (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))


def _page_geometry(page: Any, index: int) -> PageGeometry:
    displayed = page.rect
    return PageGeometry(
        index=index,
        width_pt=float(displayed.width),
        height_pt=float(displayed.height),
        width_mm=float(displayed.width) * MILLIMETRES_PER_INCH / POINTS_PER_INCH,
        height_mm=float(displayed.height) * MILLIMETRES_PER_INCH / POINTS_PER_INCH,
        rotation=int(page.rotation or 0),
        mediabox=_rect_tuple(page.mediabox),
        cropbox=_rect_tuple(page.cropbox),
    )


def _weighted_median(values: Sequence[tuple[float, float]]) -> float | None:
    cleaned = sorted(
        ((float(value), max(0.0, float(weight))) for value, weight in values if value > 0),
        key=lambda item: item[0],
    )
    if not cleaned:
        return None
    total = sum(weight for _, weight in cleaned)
    if total <= 0:
        return cleaned[len(cleaned) // 2][0]
    halfway = total / 2
    accumulated = 0.0
    for value, weight in cleaned:
        accumulated += weight
        if accumulated >= halfway:
            return value
    return cleaned[-1][0]


def _image_density_samples(page: Any) -> list[dict[str, float]]:
    samples: list[dict[str, float]] = []
    for info in page.get_image_info(hashes=False, xrefs=False):
        width_px = float(info.get("width") or 0)
        height_px = float(info.get("height") or 0)
        transform = info.get("transform")
        if width_px <= 0 or height_px <= 0 or not transform or len(transform) < 4:
            continue
        a, b, c, d = map(float, transform[:4])
        width_pt = math.hypot(a, b)
        height_pt = math.hypot(c, d)
        if width_pt <= 0 or height_pt <= 0:
            continue
        dpi_x = width_px * POINTS_PER_INCH / width_pt
        dpi_y = height_px * POINTS_PER_INCH / height_pt
        if not (10 <= dpi_x <= 2400 and 10 <= dpi_y <= 2400):
            continue
        dpi = math.sqrt(dpi_x * dpi_y)
        anisotropy = max(dpi_x, dpi_y) / max(1e-9, min(dpi_x, dpi_y))
        if anisotropy > 1.25:
            continue
        samples.append(
            {
                "dpi": dpi,
                "dpiX": dpi_x,
                "dpiY": dpi_y,
                "weight": max(1.0, width_px * height_px),
            }
        )
    return samples


def infer_pdf_raster_density(
    document: Any,
    *,
    max_pages: int = 4,
) -> dict[str, Any]:
    samples: list[dict[str, float]] = []
    for page_index in range(min(int(document.page_count), max(1, int(max_pages)))):
        samples.extend(_image_density_samples(document.load_page(page_index)))
    median = _weighted_median([(item["dpi"], item["weight"]) for item in samples])
    if median is None:
        return {
            "detected": False,
            "dpi": None,
            "pixelsPerMm": None,
            "sampleCount": 0,
            "confidence": 0.0,
        }
    total_weight = sum(item["weight"] for item in samples)
    matching_weight = sum(
        item["weight"]
        for item in samples
        if abs(item["dpi"] - median) / max(median, 1e-9) <= 0.02
    )
    return {
        "detected": True,
        "dpi": median,
        "pixelsPerMm": median / MILLIMETRES_PER_INCH,
        "sampleCount": len(samples),
        "confidence": matching_weight / max(total_weight, 1.0),
    }


def inspect_pdf(
    source_path: str | Path,
    *,
    max_density_pages: int = 4,
) -> dict[str, Any]:
    pymupdf = _pymupdf()
    path = Path(source_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with pymupdf.open(path) as document:
        if not document.is_pdf:
            raise PdfEngineError("输入文件不是 PDF")
        if document.needs_pass:
            raise PdfEngineError("暂不支持有密码的 PDF")
        ocgs = document.get_ocgs() or {}
        layers = [
            PdfLayer(
                xref=int(xref),
                name=str(detail.get("name") or f"图层 {xref}"),
                default_visible=bool(detail.get("on", True)),
                intent=tuple(str(item) for item in detail.get("intent") or ()),
                usage=str(detail.get("usage") or ""),
                order=order,
                detected_artwork=bool(
                    _ARTWORK_WORDS.search(str(detail.get("name") or ""))
                    and not _TECHNICAL_WORDS.search(str(detail.get("name") or ""))
                ),
            )
            for order, (xref, detail) in enumerate(ocgs.items())
        ]
        pages = [_page_geometry(document.load_page(index), index) for index in range(document.page_count)]
        density = infer_pdf_raster_density(document, max_pages=max_density_pages)
        direct_images = _discover_direct_layer_images(document)
        artwork_xref = choose_artwork_layer(layers)
        layer_dicts = []
        for layer in layers:
            detail = layer.as_dict()
            images = direct_images.get(layer.xref)
            detail["directImageReplacement"] = images is not None and bool(images)
            detail["imageCount"] = len(images or ())
            detail["nativePixels"] = sum(image.width * image.height for image in images or ())
            layer_dicts.append(detail)
        return {
            "kind": "pdf",
            "name": path.name,
            "size": path.stat().st_size,
            "pageCount": int(document.page_count),
            "pages": [page.as_dict() for page in pages],
            "layers": layer_dicts,
            "artworkLayerId": str(artwork_xref) if artwork_xref is not None else None,
            "detectedDpi": density["dpi"],
            "pixelsPerMm": density["pixelsPerMm"],
            "rasterDensity": density,
            "preservesOriginalObjects": bool(layers),
        }


def _layer_state(document: Any) -> dict[str, Any]:
    state = dict(document.get_layer(-1) or {})
    return {
        "on": list(state.get("on") or []),
        "off": list(state.get("off") or []),
        "basestate": state.get("basestate"),
        "rbgroups": list(state.get("rbgroups") or []),
        "locked": list(state.get("locked") or []),
    }


def _set_layer_state(document: Any, state: Mapping[str, Any]) -> None:
    document.set_layer(
        -1,
        on=list(state.get("on") or []),
        off=list(state.get("off") or []),
        basestate=state.get("basestate"),
        rbgroups=list(state.get("rbgroups") or []),
        locked=list(state.get("locked") or []),
    )


def _show_only_layer(document: Any, layer_xref: int) -> None:
    all_xrefs = [int(value) for value in (document.get_ocgs() or {}).keys()]
    if layer_xref not in all_xrefs:
        raise PdfEngineError(f"PDF 中不存在图层 {layer_xref}")
    document.set_layer(
        -1,
        on=[layer_xref],
        off=[xref for xref in all_xrefs if xref != layer_xref],
        basestate="OFF",
    )


def _bbox_close(first: Any, second: Any, tolerance: float = 0.25) -> bool:
    try:
        return all(abs(float(a) - float(b)) <= tolerance for a, b in zip(first, second))
    except (TypeError, ValueError):
        return False


def _discover_direct_layer_images(document: Any) -> dict[int, list[LayerImage] | None]:
    """Return native image replacements for image-only OCGs.

    ``get_bboxlog(layers=True)`` is the only high-level API that reports the
    effective OCG name for images inside Form XObjects. MuPDF reports image
    paint operations in the same display order as ``get_image_info``; bbox
    matching is still required so a future ordering change fails closed.
    """

    ocgs = document.get_ocgs() or {}
    name_to_xrefs: dict[str, list[int]] = {}
    for xref, detail in ocgs.items():
        name_to_xrefs.setdefault(str(detail.get("name") or ""), []).append(int(xref))
    layer_images: dict[int, list[LayerImage] | None] = {int(xref): [] for xref in ocgs}
    xref_layers: dict[int, set[int]] = {}
    unsafe_layers: set[int] = set()

    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        try:
            logs = list(page.get_bboxlog(layers=True))
        except TypeError:
            return {int(xref): None for xref in ocgs}
        image_logs = [entry for entry in logs if entry and str(entry[0]) == "fill-image"]
        infos = list(page.get_image_info(hashes=False, xrefs=True))
        image_table = {int(item[0]): int(item[1] or 0) for item in page.get_images(full=True)}
        used_logs: set[int] = set()
        layer_content_kinds: dict[int, set[str]] = {}

        for entry in logs:
            if len(entry) < 3:
                continue
            layer_name = str(entry[2] or "")
            candidates = name_to_xrefs.get(layer_name, [])
            if len(candidates) == 1:
                layer_content_kinds.setdefault(candidates[0], set()).add(str(entry[0]))

        for info in infos:
            bbox = info.get("bbox")
            matched_index = next(
                (
                    index
                    for index, entry in enumerate(image_logs)
                    if index not in used_logs
                    and len(entry) >= 3
                    and _bbox_close(entry[1], bbox)
                ),
                None,
            )
            xref = int(info.get("xref") or 0)
            if not xref or matched_index is None:
                if xref:
                    xref_layers.setdefault(xref, set()).add(0)
                continue
            used_logs.add(matched_index)
            layer_name = str(image_logs[matched_index][2] or "")
            layer_candidates = name_to_xrefs.get(layer_name, [])
            oc_xref = int(document.get_oc(xref) or 0)
            if oc_xref in ocgs:
                layer_xref = oc_xref
            elif len(layer_candidates) == 1:
                layer_xref = layer_candidates[0]
            else:
                # Replacing an image shared with unlayered/ambiguous content
                # would also recolour content outside the requested layer.
                xref_layers.setdefault(xref, set()).add(0)
                continue
            image_mask_type, image_mask_value = document.xref_get_key(xref, "ImageMask")
            mask_type, _ = document.xref_get_key(xref, "Mask")
            if (
                (image_mask_type == "bool" and image_mask_value == "true")
                or mask_type not in {"null", "unknown"}
            ):
                unsafe_layers.add(layer_xref)
            layer_images[layer_xref].append(
                LayerImage(
                    xref=xref,
                    smask=image_table.get(xref, 0),
                    width=int(info.get("width") or 0),
                    height=int(info.get("height") or 0),
                    page_index=page_index,
                    layer_xref=layer_xref,
                    layer_name=layer_name or str(ocgs[layer_xref].get("name") or layer_xref),
                    oc_xref=oc_xref,
                )
            )
            xref_layers.setdefault(xref, set()).add(layer_xref)

        for layer_xref, kinds in layer_content_kinds.items():
            if any(kind != "fill-image" for kind in kinds):
                layer_images[layer_xref] = None

    for layer_xref, images in list(layer_images.items()):
        if images is None or layer_xref in unsafe_layers:
            layer_images[layer_xref] = None
            continue
        deduplicated: dict[int, LayerImage] = {}
        for image in images:
            deduplicated.setdefault(image.xref, image)
        if any(len(xref_layers.get(xref, ())) != 1 for xref in deduplicated):
            layer_images[layer_xref] = None
        else:
            layer_images[layer_xref] = list(deduplicated.values())
    return layer_images


def render_pdf_thumbnail(
    source_path: str | Path,
    *,
    page_index: int = 0,
    max_dimension: int = 1400,
    visible_layer_xrefs: Iterable[int] | None = None,
) -> bytes:
    pymupdf = _pymupdf()
    with pymupdf.open(Path(source_path)) as document:
        if not 0 <= page_index < document.page_count:
            raise IndexError(page_index)
        original_state = _layer_state(document) if document.get_ocgs() else None
        try:
            if visible_layer_xrefs is not None and original_state is not None:
                visible = {int(value) for value in visible_layer_xrefs}
                all_xrefs = [int(value) for value in document.get_ocgs().keys()]
                document.set_layer(
                    -1,
                    on=[xref for xref in all_xrefs if xref in visible],
                    off=[xref for xref in all_xrefs if xref not in visible],
                    basestate="OFF",
                )
            page = document.load_page(page_index)
            # A lower zoom floor can turn a metre-scale drawing into a huge
            # preview. Bound by pixels, even for oversized PDF page geometry.
            scale = min(8.0, float(max_dimension) / max(page.rect.width, page.rect.height))
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(scale, scale),
                colorspace=pymupdf.csRGB,
                alpha=False,
                annots=False,
            )
            return bytes(pixmap.tobytes("png"))
        finally:
            if original_state is not None:
                _set_layer_state(document, original_state)


def plan_page_tiles(
    *,
    page_index: int,
    width_pt: float,
    height_pt: float,
    dpi: float,
    tile_size: int = DEFAULT_TILE_SIZE,
    bleed: int = DEFAULT_BLEED,
) -> list[TilePlan]:
    if not (math.isfinite(dpi) and dpi > 0):
        raise ValueError("dpi 必须大于 0")
    tile_size = max(128, min(8192, int(tile_size)))
    bleed = max(0, min(64, int(bleed)))
    width_px = max(1, round(float(width_pt) * float(dpi) / POINTS_PER_INCH))
    height_px = max(1, round(float(height_pt) * float(dpi) / POINTS_PER_INCH))
    columns = math.ceil(width_px / tile_size)
    rows = math.ceil(height_px / tile_size)
    total = rows * columns
    plans: list[TilePlan] = []
    for row in range(rows):
        for column in range(columns):
            x = column * tile_size
            y = row * tile_size
            width = min(tile_size, width_px - x)
            height = min(tile_size, height_px - y)
            outer_x = max(0, x - bleed)
            outer_y = max(0, y - bleed)
            outer_right = min(width_px, x + width + bleed)
            outer_bottom = min(height_px, y + height + bleed)
            plans.append(
                TilePlan(
                    index=len(plans),
                    total=total,
                    page_index=page_index,
                    page_width_px=width_px,
                    page_height_px=height_px,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    outer_x=outer_x,
                    outer_y=outer_y,
                    outer_width=outer_right - outer_x,
                    outer_height=outer_bottom - outer_y,
                    dpi=float(dpi),
                )
            )
    return plans


def _unpremultiply_rgba(samples: bytes, width: int, height: int, channels: int) -> bytes:
    np = _numpy()
    if channels != 4:
        raise PdfEngineError(f"预期 RGBA 渲染结果，实际为 {channels} 通道")
    data = np.frombuffer(samples, dtype=np.uint8).reshape(height, width, channels).copy()
    alpha = data[..., 3].astype(np.float32)
    visible = alpha > 0
    if np.any(visible):
        rgb = data[..., :3].astype(np.float32)
        rgb[visible] = np.minimum(255.0, np.rint(rgb[visible] * (255.0 / alpha[visible, None])))
        data[..., :3] = rgb.astype(np.uint8)
    data[~visible, :3] = 0
    return data.tobytes()


def _crop_interleaved(
    samples: bytes | bytearray | memoryview,
    *,
    source_width: int,
    source_height: int,
    channels: int,
    region: tuple[int, int, int, int],
) -> bytes:
    np = _numpy()
    x, y, width, height = region
    expected = source_width * source_height * channels
    view = memoryview(samples)
    if view.nbytes != expected:
        raise PdfEngineError(f"像素缓冲长度错误：应为 {expected}，实际为 {view.nbytes}")
    array = np.frombuffer(view, dtype=np.uint8).reshape(source_height, source_width, channels)
    return np.ascontiguousarray(array[y : y + height, x : x + width]).tobytes()


def _normalise_mapper_result(
    value: TileResult | Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> TileResult:
    result = value if isinstance(value, TileResult) else TileResult(
        cmyk=value.get("cmyk", b""),
        preview_rgba=value.get("preview_rgba") or value.get("preview"),
        stats=value.get("stats") or {},
    )
    expected = width * height * 4
    if memoryview(result.cmyk).nbytes != expected:
        raise PdfEngineError(
            f"颜色映射返回的 CMYK 长度错误：应为 {expected}，实际为 {memoryview(result.cmyk).nbytes}"
        )
    if result.preview_rgba is not None and memoryview(result.preview_rgba).nbytes != expected:
        raise PdfEngineError("颜色映射返回的预览 RGBA 长度错误")
    return result


def _load_native_pixmap(document: Any, image: LayerImage):
    pymupdf = _pymupdf()
    base = pymupdf.Pixmap(document, image.xref)
    if image.smask > 0:
        mask = pymupdf.Pixmap(document, image.smask)
        source = pymupdf.Pixmap(base, mask)
    else:
        source = pymupdf.Pixmap(base, 1)
    rgba = source if source.colorspace == pymupdf.csRGB else pymupdf.Pixmap(pymupdf.csRGB, source)
    return rgba


def _process_native_image(
    document: Any,
    image: LayerImage,
    *,
    mapper: TileMapper,
    tile_size: int,
    task_offset: int,
    total_tasks: int,
    cancelled: Callable[[], bool] | None,
    progress: Callable[[Mapping[str, Any]], None] | None,
    stats: dict[str, Any],
    temporary_directory: Path,
) -> int:
    np = _numpy()
    pixmap = _load_native_pixmap(document, image)
    width, height = int(pixmap.width), int(pixmap.height)
    # Keep the owning pixmap alive; samples_mv does not duplicate a potentially
    # hundreds-of-megabytes image. Unpremultiply only the tile sent to the GPU.
    source = np.frombuffer(pixmap.samples_mv, dtype=np.uint8).reshape(height, width, 4)
    columns = math.ceil(width / tile_size)
    rows = math.ceil(height / tile_size)
    task_count = columns * rows
    temp_path = temporary_directory / f"image-{image.xref}.cmyk"
    output = np.memmap(temp_path, mode="w+", dtype=np.uint8, shape=(height, width, 4))
    completed = 0
    try:
        for row in range(rows):
            for column in range(columns):
                _check_cancelled(cancelled)
                x = column * tile_size
                y = row * tile_size
                tile_width = min(tile_size, width - x)
                tile_height = min(tile_size, height - y)
                tile_rgba = _unpremultiply_rgba(
                    np.ascontiguousarray(source[y : y + tile_height, x : x + tile_width]).tobytes(),
                    tile_width, tile_height, 4,
                )
                plan = TilePlan(
                    index=completed,
                    total=task_count,
                    page_index=image.page_index,
                    page_width_px=width,
                    page_height_px=height,
                    x=x,
                    y=y,
                    width=tile_width,
                    height=tile_height,
                    outer_x=x,
                    outer_y=y,
                    outer_width=tile_width,
                    outer_height=tile_height,
                    dpi=0,
                )
                context = TileContext(
                    page_index=image.page_index,
                    layer_xref=image.layer_xref,
                    layer_name=image.layer_name,
                    tile=plan,
                    stats_region=(0, 0, tile_width, tile_height),
                )
                mapped = _normalise_mapper_result(
                    mapper(tile_rgba, tile_width, tile_height, context),
                    width=tile_width,
                    height=tile_height,
                )
                output[
                    y : y + tile_height,
                    x : x + tile_width,
                    :,
                ] = np.frombuffer(mapped.cmyk, dtype=np.uint8).reshape(
                    tile_height, tile_width, 4
                )
                _merge_stats(stats, mapped.stats)
                completed += 1
                if progress:
                    absolute = task_offset + completed
                    progress(
                        {
                            "phase": "pdf-native-image",
                            "completed": absolute,
                            "total": total_tasks,
                            "progress": absolute / max(1, total_tasks),
                            "page": image.page_index + 1,
                            "layer": image.layer_name,
                            "imageXref": image.xref,
                            "tile": completed,
                            "tileCount": task_count,
                        }
                    )
        output.flush()
        document.xref_set_key(image.xref, "ColorSpace", "/DeviceCMYK")
        document.xref_set_key(image.xref, "BitsPerComponent", "8")
        document.xref_set_key(image.xref, "Decode", "null")
        document.xref_set_key(image.xref, "DecodeParms", "null")
        document.xref_set_key(image.xref, "Filter", "null")
        document.update_stream(
            image.xref,
            np.ascontiguousarray(output).tobytes(),
            compress=True,
        )
        return completed
    finally:
        del output
        temp_path.unlink(missing_ok=True)


def _replace_native_layer_images(
    document: Any,
    *,
    selected: Sequence[int],
    direct_layers: Mapping[int, Sequence[LayerImage] | None],
    mapper: TileMapper,
    tile_size: int,
    cancelled: Callable[[], bool] | None,
    progress: Callable[[Mapping[str, Any]], None] | None,
    stats: dict[str, Any],
    temporary_directory: Path,
) -> dict[int, list[int]]:
    if any(not direct_layers.get(layer_xref) for layer_xref in selected):
        raise PdfEngineError("所选图层不能使用原生图像替换")
    images = [image for layer_xref in selected for image in direct_layers[layer_xref] or ()]
    total_tasks = sum(
        math.ceil(image.width / tile_size) * math.ceil(image.height / tile_size)
        for image in images
    )
    completed = 0
    replacements: dict[int, list[int]] = {layer_xref: [] for layer_xref in selected}
    for image in images:
        _check_cancelled(cancelled)
        image_tasks = _process_native_image(
            document,
            image,
            mapper=mapper,
            tile_size=tile_size,
            task_offset=completed,
            total_tasks=total_tasks,
            cancelled=cancelled,
            progress=progress,
            stats=stats,
            temporary_directory=temporary_directory,
        )
        replacements[image.layer_xref].append(image.xref)
        completed += image_tasks
    return replacements


def _render_tile_rgba(page: Any, plan: TilePlan) -> tuple[bytes, int, int]:
    pymupdf = _pymupdf()
    zoom = plan.dpi / POINTS_PER_INCH
    clip = pymupdf.Rect(
        plan.outer_x / zoom,
        plan.outer_y / zoom,
        (plan.outer_x + plan.outer_width) / zoom,
        (plan.outer_y + plan.outer_height) / zoom,
    )
    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(zoom, zoom),
        colorspace=pymupdf.csRGB,
        clip=clip,
        alpha=True,
        annots=False,
    )
    rgba = _unpremultiply_rgba(bytes(pixmap.samples), pixmap.width, pixmap.height, pixmap.n)
    return rgba, int(pixmap.width), int(pixmap.height)


def _fit_rendered_tile(
    rgba: bytes,
    rendered_width: int,
    rendered_height: int,
    plan: TilePlan,
) -> tuple[bytes, tuple[int, int, int, int]]:
    """Normalise MuPDF's edge rounding to the integer pixel tile plan."""
    np = _numpy()
    target_width, target_height = plan.outer_width, plan.outer_height
    array = np.frombuffer(rgba, dtype=np.uint8).reshape(rendered_height, rendered_width, 4)
    if abs(rendered_width - target_width) > 1 or abs(rendered_height - target_height) > 1:
        raise PdfEngineError(
            f"PDF 分块渲染尺寸异常：计划 {target_width}x{target_height}，实际 {rendered_width}x{rendered_height}"
        )
    if rendered_width != target_width or rendered_height != target_height:
        normalised = np.zeros((target_height, target_width, 4), dtype=np.uint8)
        copy_height = min(rendered_height, target_height)
        copy_width = min(rendered_width, target_width)
        normalised[:copy_height, :copy_width] = array[:copy_height, :copy_width]
        array = normalised
    return np.ascontiguousarray(array).tobytes(), plan.core_in_outer


def _insert_cmyk_tile(
    page: Any,
    *,
    cmyk: bytes,
    alpha: bytes,
    width: int,
    height: int,
    rect: Any,
    ocg_xref: int,
) -> int:
    pymupdf = _pymupdf()
    np = _numpy()
    cmyk_array = np.frombuffer(cmyk, dtype=np.uint8).reshape(height, width, 4)
    alpha_array = np.frombuffer(alpha, dtype=np.uint8).reshape(height, width)
    color_pixmap = pymupdf.Pixmap(
        pymupdf.csCMYK,
        width,
        height,
        np.ascontiguousarray(cmyk_array).tobytes(),
        False,
    )
    pixmap = pymupdf.Pixmap(color_pixmap, 1)
    pixmap.set_alpha(np.ascontiguousarray(alpha_array).tobytes(), premultiply=0)
    image_xref = int(
        page.insert_image(
            rect,
            pixmap=pixmap,
            keep_proportion=False,
            overlay=False,
            oc=ocg_xref,
        )
    )
    # insert_image can leave raw pixel streams in the document until save.
    # Compress each tile now so a large drawing does not accumulate all raw
    # CMYK and alpha pixels in RAM. The image stream uses straight CMYK;
    # transparency is represented separately by its soft mask.
    document = page.parent
    document.xref_set_key(image_xref, "ColorSpace", "/DeviceCMYK")
    document.xref_set_key(image_xref, "BitsPerComponent", "8")
    document.xref_set_key(image_xref, "Decode", "null")
    document.xref_set_key(image_xref, "DecodeParms", "null")
    document.update_stream(image_xref, cmyk, compress=True)
    mask_type, mask_value = document.xref_get_key(image_xref, "SMask")
    if mask_type == "xref":
        mask_xref = int(mask_value.split()[0])
        # MuPDF may optimise binary masks to one bit per pixel. We are writing
        # the original eight-bit alpha bytes, so update its dictionary too.
        document.xref_set_key(mask_xref, "BitsPerComponent", "8")
        document.xref_set_key(mask_xref, "Decode", "null")
        document.xref_set_key(mask_xref, "DecodeParms", "null")
        document.update_stream(mask_xref, alpha, compress=True)
    return image_xref


def _merge_stats(total: dict[str, Any], addition: Mapping[str, Any]) -> None:
    for key, value in addition.items():
        if not isinstance(value, (int, float)):
            continue
        if key in {"maxDeltaE", "max_delta_e"}:
            total[key] = max(float(total.get(key, 0)), float(value))
        else:
            total[key] = float(total.get(key, 0)) + float(value)


def _pdf_parse_configuration():
    from pypdf import apply_configuration

    # These are content-token/object limits, not image decompression limits.
    # Larger plotter PDFs can legitimately contain millions of path operands.
    return apply_configuration(
        stream_parsing_work_maximum_count=max(1, int(os.getenv("CMYK_PDF_PARSE_MAX_TOKENS", "20000000"))),
        array_maximum_element_count=1_000_000,
        dictionary_maximum_element_count=1_000_000,
    )


def _isolate_render_layer(document: Any, selected_xref: int) -> None:
    """Remove unrelated paint from a disposable rendering document.

    Merely switching OCG visibility leaves unlayered artwork visible. Parse
    content streams (not image pixels) to retain state/clip/transform operators
    and paint only the selected OCG, including OCGs nested in Form XObjects.
    The separate output document is never modified by this filtering.
    """
    try:
        from pypdf.generic import (
            ArrayObject, ContentStream, DecodedStreamObject, DictionaryObject,
            IndirectObject, NameObject, NumberObject, read_object,
        )
    except ImportError as exc:
        raise PdfEngineUnavailable("PDF 图层隔离需要 pypdf") from exc

    class ObjectResolver:
        # Supply pypdf with lazy object resolution backed by MuPDF. Unlike a
        # PdfReader(path), this never reads the complete large PDF into BytesIO.
        strict = True

        def get_object(self, reference):
            return parse_object(reference.idnum)

    resolver = ObjectResolver()

    def parse_object(xref: int):
        with _pdf_parse_configuration():
            return read_object(BytesIO(document.xref_object(xref, compressed=False).encode("utf-8")), resolver)

    def resolve(value):
        return parse_object(value.idnum) if isinstance(value, IndirectObject) else value

    def as_dict(value):
        value = resolve(value)
        return DictionaryObject(value) if isinstance(value, DictionaryObject) else DictionaryObject()

    def serialise(value) -> str:
        buffer = BytesIO()
        value.write_to_stream(buffer)
        return buffer.getvalue().decode("latin-1")

    def oc_state(value) -> tuple[bool, bool]:
        """Return visible-with-selected-only, references-selected."""
        if isinstance(value, IndirectObject):
            if value.idnum in (document.get_ocgs() or {}):
                return value.idnum == selected_xref, value.idnum == selected_xref
        detail = as_dict(value)
        if detail.get("/Type") == "/OCG":
            # OCGs normally arrive as references; direct OCG dictionaries are
            # ambiguous, so fail instead of mapping unrelated artwork.
            raise PdfEngineError("不支持直接嵌入的 OCG 字典")
        expression = detail.get("/VE")
        if expression:
            def evaluate(expr):
                if not isinstance(expr, ArrayObject):
                    return oc_state(expr)
                children = [evaluate(child) for child in expr[1:]]
                operator = str(expr[0])
                if operator == "/Not" and len(children) == 1:
                    return not children[0][0], children[0][1]
                if operator not in {"/And", "/Or"}:
                    raise PdfEngineError("不支持的 PDF 可选内容可见性表达式")
                return (all(state[0] for state in children) if operator == "/And" else any(state[0] for state in children), any(state[1] for state in children))
            return evaluate(expression)
        groups = detail.get("/OCGs")
        groups = groups if isinstance(groups, ArrayObject) else [groups] if groups else []
        states = [oc_state(group) for group in groups]
        policy = str(detail.get("/P", "/AnyOn"))
        values = [state[0] for state in states]
        if policy == "/AllOn":
            visible = all(values)
        elif policy == "/AnyOff":
            visible = any(not item for item in values)
        elif policy == "/AllOff":
            visible = all(not item for item in values)
        elif policy == "/AnyOn":
            visible = any(values)
        else:
            raise PdfEngineError(f"不支持的 PDF 可选内容策略：{policy}")
        return visible, any(state[1] for state in states)

    forms: dict[tuple[int, bool], int] = {}
    visiting: set[tuple[int, bool]] = set()

    def filter_stream(data: bytes, resources, inherited: bool):
        resources = as_dict(resources)
        xobjects = as_dict(resources.get("/XObject"))
        properties = as_dict(resources.get("/Properties"))
        stream = DecodedStreamObject()
        stream.set_data(data)
        content = ContentStream(stream, resolver)
        with _pdf_parse_configuration():
            source_operations = content.operations
        operations = []
        active, visible = inherited, True
        marked_stack: list[tuple[bool, bool, bool]] = []
        text_mode = 0
        saved_text_modes: list[int] = []
        for operands, operator in source_operations:
            if operator in {b"BDC", b"BMC"}:
                optional_content = operator == b"BDC" and operands and str(operands[0]) == "/OC"
                marked_stack.append((active, visible, bool(optional_content)))
                if optional_content:
                    property_value = operands[1]
                    if isinstance(property_value, NameObject):
                        property_value = properties.get(property_value)
                    enabled, references_selected = oc_state(property_value)
                    active = active or references_selected
                    visible = visible and enabled
                    # Visibility has already been evaluated. Strip this OCG
                    # wrapper so originally hidden selected layers also render.
                    continue
            elif operator == b"EMC":
                if not marked_stack:
                    raise PdfEngineError("PDF 标记内容栈不匹配")
                active, visible, optional_content = marked_stack.pop()
                if optional_content:
                    continue
            elif operator == b"q":
                saved_text_modes.append(text_mode)
            elif operator == b"Q" and saved_text_modes:
                text_mode = saved_text_modes.pop()
            elif operator == b"Tr":
                text_mode = int(operands[0])
            elif operator == b"Do":
                reference = xobjects.get(operands[0])
                if not isinstance(reference, IndirectObject):
                    raise PdfEngineError("PDF XObject 引用无效")
                obj = as_dict(reference)
                own_active, own_visible = active, visible
                if obj.get("/OC"):
                    enabled, references_selected = oc_state(obj.get("/OC"))
                    own_active = active or references_selected
                    own_visible = visible and enabled
                if not own_visible:
                    continue
                if str(obj.get("/Subtype")) == "/Form":
                    key = (reference.idnum, own_active)
                    if key in visiting:
                        raise PdfEngineError("PDF Form XObject 存在递归引用")
                    if key not in forms:
                        visiting.add(key)
                        child_data, child_resources = filter_stream(
                            document.xref_stream(reference.idnum),
                            obj.get("/Resources", resources), own_active,
                        )
                        clone_xref = document.get_new_xref()
                        obj[NameObject("/Resources")] = child_resources
                        if "/OC" in obj:
                            del obj["/OC"]
                        document.update_object(clone_xref, serialise(obj))
                        document.update_stream(clone_xref, child_data, compress=True)
                        forms[key] = clone_xref
                        visiting.remove(key)
                    alias = NameObject(f"/CMYKLayerForm{forms[key]}")
                    xobjects[alias] = IndirectObject(forms[key], 0, resolver)
                    operands = [alias]
                elif not own_active:
                    continue
                elif obj.get("/OC"):
                    document.xref_set_key(reference.idnum, "OC", "null")
            elif not (active and visible):
                if operator in {b"S", b"s", b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*"}:
                    operations.append(([], b"n"))
                    continue
                if operator in {b"sh", b"INLINE IMAGE"}:
                    continue
                if operator in {b"Tj", b"TJ", b"'", b'"'}:
                    # Invisible text still advances the text matrix, preserving
                    # positions when marked content changes inside BT ... ET.
                    operations.append(([NumberObject(7 if text_mode >= 4 else 3)], b"Tr"))
                    operations.append((operands, operator))
                    operations.append(([NumberObject(text_mode)], b"Tr"))
                    continue
            operations.append((operands, operator))
        if marked_stack:
            raise PdfEngineError("PDF 标记内容栈不匹配")
        content.operations = operations
        resources[NameObject("/XObject")] = xobjects
        return content.get_data(), resources

    for page in document:
        # Resources can be inherited through the PDF page tree.
        owner = page.xref
        resources = None
        while owner:
            detail = as_dict(parse_object(owner))
            resources = detail.get("/Resources")
            if resources is not None:
                break
            parent = detail.get("/Parent")
            owner = parent.idnum if isinstance(parent, IndirectObject) else 0
        data = b"\n".join(document.xref_stream(xref) for xref in page.get_contents())
        filtered, filtered_resources = filter_stream(data, resources, False)
        stream_xref = document.get_new_xref()
        document.update_object(stream_xref, "<<>>")
        document.update_stream(stream_xref, filtered, compress=True)
        document.xref_set_key(page.xref, "Resources", serialise(filtered_resources))
        page.set_contents(stream_xref)
    _show_only_layer(document, selected_xref)


def _overlay_insertion_anchors(document: Any, selected: Sequence[int]) -> dict[tuple[int, int], int | None]:
    """Locate safe stream boundaries that preserve each layer's paint order.

    Whole-layer rasterisation cannot preserve interleaved paint operations.
    Reject those layouts instead of silently moving selected artwork below
    backgrounds or above technical lines. Native image replacement does not
    need this restriction.
    """
    from pypdf.generic import ContentStream, DecodedStreamObject

    pymupdf = _pymupdf()
    anchors: dict[tuple[int, int], int | None] = {}
    with pymupdf.open(document.name) as scan:
        ocgs = scan.get_ocgs() or {}
        names = {xref: str(ocgs[xref].get("name") or "") for xref in selected}
        for config in scan.layer_ui_configs():
            if config.get("type") != "label":
                scan.set_layer_ui_config(config["number"], action=0)
        def paint_names(page):
            entries = list(page.get_bboxlog(layers=True))
            images = list(page.get_image_info(xrefs=True))
            used: set[int] = set()
            result = []
            for entry in entries:
                if len(entry) < 3 or entry[0] == "ignore-text":
                    continue
                name = str(entry[2] or "")
                if entry[0] == "fill-image":
                    match = next((index for index, image in enumerate(images) if index not in used and _bbox_close(entry[1], image.get("bbox"))), None)
                    if match is not None:
                        used.add(match)
                        image_xref = int(images[match].get("xref") or 0)
                        attached = int(scan.get_oc(image_xref) or 0) if image_xref else 0
                        if attached in ocgs:
                            name = str(ocgs[attached].get("name") or "")
                result.append(name)
            return result
        for page in scan:
            contents = page.get_contents()
            full_names = paint_names(page)
            per_stream: list[list[str]] = []
            safe_boundary = True
            for stream_xref in contents:
                data = scan.xref_stream(stream_xref)
                parsed = DecodedStreamObject()
                parsed.set_data(data)
                depth, marked = 0, 0
                try:
                    with _pdf_parse_configuration():
                        operators = ContentStream(parsed, None).operations
                    for _, operator in operators:
                        if operator == b"q":
                            depth += 1
                        elif operator == b"Q":
                            depth -= 1
                        elif operator in {b"BMC", b"BDC"}:
                            marked += 1
                        elif operator == b"EMC":
                            marked -= 1
                        elif operator in {b"cm", b"W", b"W*"} and depth == 0:
                            safe_boundary = False
                        if depth < 0 or marked < 0:
                            safe_boundary = False
                    if depth or marked:
                        safe_boundary = False
                except Exception as exc:
                    raise PdfEngineError("无法安全解析 PDF 内容流；请保留原密度使用原生图像替换，或重新导出各图层独立的内容流") from exc
                page.set_contents(stream_xref)
                per_stream.append(paint_names(page))
            for layer_xref, layer_name in names.items():
                matching = [index for index, entries in enumerate(per_stream) if layer_name in entries]
                positions = [index for index, name in enumerate(full_names) if name == layer_name]
                mixed_stream = any(any(name != layer_name for name in per_stream[index]) for index in matching)
                interleaved = bool(positions) and any(name != layer_name for name in full_names[min(positions):max(positions) + 1])
                scope_mismatch = sum(entries.count(layer_name) for entries in per_stream) != len(positions)
                if positions and (not safe_boundary or mixed_stream or interleaved or scope_mismatch):
                    raise PdfEngineError(
                        f"第 {page.number + 1} 页的图层“{layer_name}”与其他内容交错，无法在重采样时保证层叠顺序；"
                        "请重新导出各图层独立的内容流，或省略 dpi 保留原密度以使用原生图像替换"
                    )
                anchors[(page.number, layer_xref)] = contents[max(matching)] if matching else None
            scan.xref_set_key(page.xref, "Contents", "[" + " ".join(f"{xref} 0 R" for xref in contents) + "]")
    return anchors


def _replace_overlay_layers(
    render_document: Any,
    output_document: Any,
    *,
    selected: Sequence[int],
    source_ocgs: Mapping[int, Mapping[str, Any]],
    dpi: float,
    mapper: TileMapper,
    tile_size: int,
    bleed: int,
    cancelled: Callable[[], bool] | None,
    progress: Callable[[Mapping[str, Any]], None] | None,
) -> tuple[dict[int, int], list[list[TilePlan]], dict[str, Any], list[str]]:
    pymupdf = _pymupdf()
    original_render_state = _layer_state(render_document)
    anchors = _overlay_insertion_anchors(render_document, selected)
    replacements: dict[int, int] = {}
    for xref in selected:
        name = str(source_ocgs[xref].get("name") or f"图层 {xref}")
        replacements[xref] = int(
            output_document.add_ocg(f"{name}（已处理）", on=True, usage="Artwork")
        )

    page_plans: list[list[TilePlan]] = []
    for page_index in range(render_document.page_count):
        page = render_document.load_page(page_index)
        original_rotation = int(page.rotation or 0)
        if original_rotation:
            page.set_rotation(0)
        page_plans.append(
            plan_page_tiles(
                page_index=page_index,
                width_pt=float(page.cropbox.width),
                height_pt=float(page.cropbox.height),
                dpi=dpi,
                tile_size=tile_size,
                bleed=bleed,
            )
        )
        if original_rotation:
            page.set_rotation(original_rotation)

    total_tiles = sum(len(plans) for plans in page_plans) * len(selected)
    completed_tiles = 0
    stats: dict[str, Any] = {}
    processed_layer_names: list[str] = []
    isolated_render = None
    try:
        for layer_xref in selected:
            _check_cancelled(cancelled)
            if isolated_render is not None:
                isolated_render.close()
            isolated_render = pymupdf.open(render_document.name)
            _isolate_render_layer(isolated_render, layer_xref)
            layer_name = str(source_ocgs[layer_xref].get("name") or f"图层 {layer_xref}")
            processed_layer_names.append(layer_name)
            replacement_xref = replacements[layer_xref]
            for page_index, plans in enumerate(page_plans):
                _check_cancelled(cancelled)
                render_page = isolated_render.load_page(page_index)
                original_rotation = int(render_page.rotation or 0)
                if original_rotation:
                    render_page.set_rotation(0)
                output_page = output_document.load_page(page_index)
                previous_contents = output_page.get_contents()
                for plan in plans:
                    _check_cancelled(cancelled)
                    rgba, rendered_width, rendered_height = _render_tile_rgba(render_page, plan)
                    rgba, core_region = _fit_rendered_tile(rgba, rendered_width, rendered_height, plan)
                    np = _numpy()
                    rgba_array = np.frombuffer(rgba, dtype=np.uint8).reshape(
                        plan.outer_height, plan.outer_width, 4
                    )
                    alpha = rgba_array[..., 3]
                    core_x, core_y, core_width, core_height = core_region
                    core_alpha_array = alpha[
                        core_y : core_y + core_height,
                        core_x : core_x + core_width,
                    ]
                    if np.any(core_alpha_array):
                        context = TileContext(
                            page_index=page_index,
                            layer_xref=layer_xref,
                            layer_name=layer_name,
                            tile=plan,
                            stats_region=core_region,
                        )
                        mapped = _normalise_mapper_result(
                            mapper(rgba, plan.outer_width, plan.outer_height, context),
                            width=plan.outer_width,
                            height=plan.outer_height,
                        )
                        core_cmyk = _crop_interleaved(
                            mapped.cmyk,
                            source_width=plan.outer_width,
                            source_height=plan.outer_height,
                            channels=4,
                            region=core_region,
                        )
                        core_alpha = np.ascontiguousarray(core_alpha_array).tobytes()
                        zoom = plan.dpi / POINTS_PER_INCH
                        rect = pymupdf.Rect(
                            plan.x / zoom,
                            plan.y / zoom,
                            (plan.x + plan.width) / zoom,
                            (plan.y + plan.height) / zoom,
                        )
                        _insert_cmyk_tile(
                            output_page,
                            cmyk=core_cmyk,
                            alpha=core_alpha,
                            width=plan.width,
                            height=plan.height,
                            rect=rect,
                            ocg_xref=replacement_xref,
                        )
                        _merge_stats(stats, mapped.stats)
                    completed_tiles += 1
                    if progress:
                        progress(
                            {
                                "phase": "pdf-tile",
                                "completed": completed_tiles,
                                "total": total_tiles,
                                "progress": completed_tiles / max(1, total_tiles),
                                "page": page_index + 1,
                                "pageCount": render_document.page_count,
                                "layer": layer_name,
                                "tile": plan.index + 1,
                                "tileCount": plan.total,
                            }
                        )
                current_contents = output_page.get_contents()
                generated = [xref for xref in current_contents if xref not in previous_contents]
                if generated:
                    anchor = anchors[(page_index, layer_xref)]
                    if anchor is None:
                        raise PdfEngineError("PDF 选层渲染产生了无法定位层叠顺序的内容")
                    ordered = [xref for xref in current_contents if xref not in generated]
                    insertion_index = ordered.index(anchor) + 1
                    ordered[insertion_index:insertion_index] = generated
                    output_document.xref_set_key(output_page.xref, "Contents", "[" + " ".join(f"{xref} 0 R" for xref in ordered) + "]")
                if original_rotation:
                    render_page.set_rotation(original_rotation)
    finally:
        if isolated_render is not None:
            isolated_render.close()
        _set_layer_state(render_document, original_render_state)
    return replacements, page_plans, stats, processed_layer_names


def replace_pdf_layers(
    source_path: str | Path,
    output_path: str | Path,
    *,
    selected_layer_xrefs: Iterable[int | str],
    dpi: float,
    mapper: TileMapper,
    preserve_native_density: bool = True,
    tile_size: int = DEFAULT_TILE_SIZE,
    bleed: int = DEFAULT_BLEED,
    cancelled: Callable[[], bool] | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Replace selected OCG layers while preserving every unselected PDF object.

    Image-only OCGs use native image replacement, preserving each image xref,
    transform, bounding box and pixel dimensions. Other OCGs use a seam-safe
    raster overlay: the selected original OCG defaults to OFF and a mapped OCG
    defaults to ON. Unselected layers are never rendered or rebuilt.
    """

    pymupdf = _pymupdf()
    source = Path(source_path)
    output = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.resolve() == output.resolve():
        raise PdfEngineError("输出路径不能覆盖输入 PDF")
    if output.exists():
        raise FileExistsError(output)
    if not (math.isfinite(dpi) and dpi > 0):
        raise PdfEngineError("输出 DPI 必须大于 0")
    tile_size = max(128, min(8192, int(tile_size)))

    selected = []
    for value in selected_layer_xrefs:
        xref = int(value)
        if xref not in selected:
            selected.append(xref)
    if not selected:
        raise PdfEngineError("请至少选择一个要处理的 PDF 图层")

    output.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(source) as render_document, pymupdf.open(source) as output_document:
        if not render_document.is_pdf or not output_document.is_pdf:
            raise PdfEngineError("输入文件不是 PDF")
        if render_document.needs_pass:
            raise PdfEngineError("暂不支持有密码的 PDF")
        source_ocgs = render_document.get_ocgs() or {}
        missing = [xref for xref in selected if xref not in source_ocgs]
        if missing:
            raise PdfEngineError(f"PDF 中不存在所选图层：{missing}")

        source_xref_length = output_document.xref_length()
        source_pages = [
            _page_geometry(output_document.load_page(index), index)
            for index in range(output_document.page_count)
        ]
        page_plans = [
            plan_page_tiles(
                page_index=page.index,
                width_pt=float(output_document.load_page(page.index).cropbox.width),
                height_pt=float(output_document.load_page(page.index).cropbox.height),
                dpi=dpi,
                tile_size=tile_size,
                bleed=bleed,
            )
            for page in source_pages
        ]
        stats: dict[str, Any] = {}
        processed_layer_names = [
            str(source_ocgs[xref].get("name") or f"图层 {xref}") for xref in selected
        ]
        direct_layers = _discover_direct_layer_images(output_document)
        # Native replacement preserves the original pixel dimensions. An
        # explicit density request must take the rendering path instead.
        # Keep huge images out of the native path, whose PDF stream API must
        # materialise one complete image at the final write step.
        native_max_pixels = max(1, int(os.getenv("CMYK_PDF_NATIVE_MAX_PIXELS", "64000000")))
        direct_mode = preserve_native_density and all(
            direct_layers.get(xref)
            and all(image.width * image.height <= native_max_pixels for image in direct_layers[xref] or ())
            for xref in selected
        )
        overlay_replacements: dict[int, int] = {}
        native_replacements: dict[int, list[int]] = {}
        if direct_mode:
            with tempfile.TemporaryDirectory(
                prefix="cmyk-pdf-", dir=str(output.parent)
            ) as temp_dir:
                native_replacements = _replace_native_layer_images(
                    output_document,
                    selected=selected,
                    direct_layers=direct_layers,
                    mapper=mapper,
                    tile_size=tile_size,
                    cancelled=cancelled,
                    progress=progress,
                    stats=stats,
                    temporary_directory=Path(temp_dir),
                )
            replacement_mode = "native-images"
        else:
            overlay_replacements, page_plans, stats, processed_layer_names = _replace_overlay_layers(
                render_document,
                output_document,
                selected=selected,
                source_ocgs=source_ocgs,
                dpi=dpi,
                mapper=mapper,
                tile_size=tile_size,
                bleed=bleed,
                cancelled=cancelled,
                progress=progress,
            )
            replacement_mode = "ocg-overlay"

        final_ocgs = output_document.get_ocgs() or {}
        final_on = [int(xref) for xref, detail in final_ocgs.items() if bool(detail.get("on", True))]
        if replacement_mode == "ocg-overlay":
            final_on = [xref for xref in final_on if xref not in selected]
            for replacement in overlay_replacements.values():
                if replacement not in final_on:
                    final_on.append(replacement)
        final_off = [int(xref) for xref in final_ocgs if int(xref) not in final_on]
        output_document.set_layer(
            -1,
            on=final_on,
            off=final_off,
            basestate="OFF",
        )

        _check_cancelled(cancelled)
        if progress:
            progress({"phase": "pdf-save", "progress": 0.99, "label": "写入最终 PDF"})
        output_document.save(
            output,
            garbage=0,
            deflate=True,
            clean=False,
        )

    with pymupdf.open(output) as verification:
        output_pages = [_page_geometry(verification.load_page(i), i) for i in range(verification.page_count)]
        output_layers = verification.get_ocgs() or {}
        if len(output_pages) != len(source_pages):
            raise PdfEngineError("输出 PDF 页数发生变化")
        for before, after in zip(source_pages, output_pages):
            if (
                abs(before.width_pt - after.width_pt) > 0.001
                or abs(before.height_pt - after.height_pt) > 0.001
                or before.rotation != after.rotation
            ):
                raise PdfEngineError("输出 PDF 页面尺寸或方向发生变化")
        if replacement_mode == "ocg-overlay":
            for replacement in overlay_replacements.values():
                if replacement not in output_layers or not output_layers[replacement].get("on", False):
                    raise PdfEngineError("输出 PDF 的处理图层未正确写入")
            for original in selected:
                if original not in output_layers or output_layers[original].get("on", True):
                    raise PdfEngineError("输出 PDF 的原始待处理图层未正确隐藏")
        else:
            for original in selected:
                if original not in output_layers:
                    raise PdfEngineError("原始 PDF 图层在原生图像替换后丢失")
            for image_xrefs in native_replacements.values():
                for image_xref in image_xrefs:
                    if verification.xref_get_key(image_xref, "ColorSpace") != (
                        "name",
                        "/DeviceCMYK",
                    ):
                        raise PdfEngineError("原生图像未写成无 ICC 的 DeviceCMYK")
                    if verification.xref_get_key(image_xref, "BitsPerComponent") != (
                        "int",
                        "8",
                    ):
                        raise PdfEngineError("原生图像不是 8-bit CMYK")
            if verification.xref_length() != source_xref_length:
                raise PdfEngineError("原生图像替换意外新增了 PDF 对象")

    return {
        "path": str(output),
        "byteLength": output.stat().st_size,
        "pageCount": len(output_pages),
        "pages": [page.as_dict() for page in output_pages],
        "dpi": float(dpi),
        "pixelsPerMm": float(dpi) / MILLIMETRES_PER_INCH,
        "outputPages": [
            {
                "pixelWidth": plans[0].page_width_px if plans else 0,
                "pixelHeight": plans[0].page_height_px if plans else 0,
            }
            for plans in page_plans
        ],
        "processedLayerNames": processed_layer_names,
        "replacementMode": replacement_mode,
        "replacementLayers": {
            str(source_xref): target_xref
            for source_xref, target_xref in overlay_replacements.items()
        },
        "replacedImageXrefs": {
            str(layer_xref): image_xrefs
            for layer_xref, image_xrefs in native_replacements.items()
        },
        "stats": stats,
        "unselectedObjectsPreserved": True,
    }

