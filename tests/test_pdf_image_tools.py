from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import pymupdf
from PIL import Image

from api.pdf_engine import PdfProcessingCancelled
from api.pdf_image_tools import PdfImageToolError, place_image_on_pdf


MM_TO_PT = 72 / 25.4


def _quadrants(path: Path, *, alpha: bool = False) -> np.ndarray:
    pixels = np.zeros((40, 80, 4 if alpha else 3), dtype=np.uint8)
    pixels[:20, :40, :3] = (255, 0, 0)
    pixels[:20, 40:, :3] = (0, 255, 0)
    pixels[20:, :40, :3] = (0, 0, 255)
    pixels[20:, 40:, :3] = (255, 255, 0)
    if alpha:
        pixels[..., 3] = 255
        pixels[:20, :40, 3] = 0
        pixels[20:, 40:, 3] = 128
    with Image.fromarray(pixels) as image:
        image.save(path)
    return pixels


def _source(path: Path, *, rotation=0, user_unit=1, crop=False, page_count=1):
    document = pymupdf.open()
    layer = document.add_ocg("Original")
    for index in range(page_count):
        page = document.new_page(width=360, height=280)
        if crop:
            page.set_cropbox(pymupdf.Rect(40, 30, 320, 250))
        if user_unit != 1:
            document.xref_set_key(page.xref, "UserUnit", str(user_unit))
            page = document.reload_page(page)
        page.draw_rect(page.rect, color=None, fill=(0.8, 0.8, 0.8), oc=layer)
        old_pixels = np.full((7, 13, 3), 80, dtype=np.uint8)
        page.insert_image(pymupdf.Rect(1, 1, 26, 15), pixmap=pymupdf.Pixmap(pymupdf.csRGB, 13, 7, old_pixels.tobytes(), False), oc=layer)
        page.set_rotation(rotation)
    document.save(path, deflate=True)
    document.close()


def _place(source, image, output, **changes):
    options = dict(x_mm=13, y_mm=17, width_mm=40, height_mm=25, layer_name="新增图像")
    options.update(changes)
    return place_image_on_pdf(source, image, output, **options)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("user_unit", [1, 2])
@pytest.mark.parametrize("crop", [False, True])
def test_display_cropbox_coordinates_and_upright_corners(tmp_path, rotation, user_unit, crop):
    source, image, output = tmp_path / "source.pdf", tmp_path / "image.png", tmp_path / "output.pdf"
    _source(source, rotation=rotation, user_unit=user_unit, crop=crop)
    _quadrants(image)
    source_bytes = source.read_bytes()
    with pymupdf.open(source) as original:
        old_image = original[0].get_images(full=True)[0][0]
        original_pixels = original.xref_stream(old_image)
        original_dict = original.xref_object(old_image)
        original_contents = {xref: original.xref_stream(xref) for xref in original[0].get_contents()}
        original_rect = original[0].rect
        original_crop = original[0].cropbox
        original_ocgs = original.get_ocgs()
    result = _place(source, image, output)
    assert output.read_bytes().startswith(source_bytes)  # incremental update
    assert source.read_bytes() == source_bytes
    assert result["imagePixels"]["width"] == 80 and result["imagePixels"]["height"] == 40
    assert result["placementMm"] == {"x": 13, "y": 17, "width": 40, "height": 25}
    with pymupdf.open(output) as final:
        page = final[0]
        assert page.rotation == rotation and page.rect == original_rect and page.cropbox == original_crop
        assert final.xref_stream(old_image) == original_pixels
        assert final.xref_object(old_image) == original_dict
        for xref, content in original_contents.items():
            assert final.xref_stream(xref) == content
        for xref, detail in original_ocgs.items():
            assert final.get_ocgs()[xref] == detail
        assert final.get_ocgs()[int(result["newLayerId"])]["on"]
        assert final.get_oc(result["imageXref"]) == int(result["newLayerId"])
        assert final.xref_get_key(result["imageXref"], "Width") == ("int", "80")
        assert final.xref_get_key(result["imageXref"], "Height") == ("int", "40")
        pixmap = page.get_pixmap(colorspace=pymupdf.csRGB)
        rendered = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
        for fx, fy, expected in ((0.25, 0.25, (255, 0, 0)), (0.75, 0.25, (0, 255, 0)), (0.25, 0.75, (0, 0, 255)), (0.75, 0.75, (255, 255, 0))):
            px = round((13 + 40 * fx) * MM_TO_PT)
            py = round((17 + 25 * fy) * MM_TO_PT)
            assert rendered[py, px].tolist() == pytest.approx(expected, abs=3)
        outside_x, outside_y = round(12 * MM_TO_PT), round(25 * MM_TO_PT)
        assert rendered[outside_y, outside_x].tolist() == pytest.approx((204, 204, 204), abs=3)


def test_preserves_alpha_and_stretches_exact_dimensions(tmp_path):
    source, image, output = tmp_path / "source.pdf", tmp_path / "alpha.png", tmp_path / "output.pdf"
    _source(source)
    pixels = _quadrants(image, alpha=True)
    result = _place(source, image, output, width_mm=40, height_mm=40)
    assert result["imagePixels"]["hasAlpha"]
    with pymupdf.open(output) as document:
        image_xref = result["imageXref"]
        smask = int(document.xref_get_key(image_xref, "SMask")[1].split()[0])
        mask = pymupdf.Pixmap(document, smask)
        assert np.frombuffer(mask.samples, dtype=np.uint8).reshape(40, 80).tolist() == pixels[..., 3].tolist()
        pixmap = document[0].get_pixmap(colorspace=pymupdf.csRGB)
        rendered = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
        transparent = rendered[round(27 * MM_TO_PT), round(23 * MM_TO_PT)]
        opaque = rendered[round(27 * MM_TO_PT), round(43 * MM_TO_PT)]
        semi = rendered[round(47 * MM_TO_PT), round(43 * MM_TO_PT)]
        assert transparent.tolist() == pytest.approx((204, 204, 204), abs=3)
        assert opaque.tolist() == pytest.approx((0, 255, 0), abs=3)
        assert 90 < semi[2] < 115 and semi[0] > 220 and semi[1] > 220


def test_always_new_independent_same_named_layer_and_keeps_old_state(tmp_path):
    source, image = tmp_path / "source.pdf", tmp_path / "image.png"
    first, second = tmp_path / "first.pdf", tmp_path / "second.pdf"
    _quadrants(image)
    document = pymupdf.open()
    page = document.new_page(width=360, height=280)
    existing = document.add_ocg("同名层", on=False)
    old_image = page.insert_image(pymupdf.Rect(0, 0, 100, 50), filename=image, oc=existing)
    document.set_layer(-1, on=[], off=[existing], basestate="OFF")
    document.save(source)
    document.close()
    one = _place(source, image, first, layer_name="同名层")
    two = _place(first, image, second, layer_name="同名层", x_mm=60)
    assert one["newLayerId"] != two["newLayerId"]
    with pymupdf.open(second) as final:
        ocgs = final.get_ocgs()
        assert len(ocgs) == 3
        assert all(detail["name"] == "同名层" for detail in ocgs.values())
        assert not ocgs[existing]["on"]
        assert ocgs[int(one["newLayerId"])]["on"] and ocgs[int(two["newLayerId"])]["on"]
        assert final.get_oc(old_image) == existing
        assert final.get_oc(one["imageXref"]) == int(one["newLayerId"])
        assert final.get_oc(two["imageXref"]) == int(two["newLayerId"])
        assert final.xref_get_key(int(one["newLayerId"]), "CMYKRecipeOCGId") != final.xref_get_key(int(two["newLayerId"]), "CMYKRecipeOCGId")


def test_one_based_target_page_leaves_other_page_content(tmp_path):
    source, image, output = tmp_path / "source.pdf", tmp_path / "image.png", tmp_path / "output.pdf"
    _source(source, page_count=2)
    _quadrants(image)
    with pymupdf.open(source) as original:
        before = original[0].get_pixmap().tobytes("png")
    result = _place(source, image, output, page_number=2)
    assert result["pageNumber"] == 2 and result["pageCount"] == 2
    with pymupdf.open(output) as final:
        assert final[0].get_pixmap().tobytes("png") == before
        assert any(item[0] == result["imageXref"] for item in final[1].get_images())


@pytest.mark.parametrize("extension", ["png", "jpg", "tif", "bmp", "webp"])
def test_common_raster_formats_without_pdf_rasterization(tmp_path, extension):
    source, image, output = tmp_path / "source.pdf", tmp_path / f"image.{extension}", tmp_path / "output.pdf"
    _source(source)
    _quadrants(image)
    result = _place(source, image, output)
    assert result["imagePixels"]["width"] == 80 and result["imagePixels"]["height"] == 40
    with pymupdf.open(output) as final:
        assert len(final[0].get_images()) == 2
        if extension == "jpg":
            assert final.xref_get_key(result["imageXref"], "Filter") == ("name", "/DCTDecode")
            assert final.xref_stream_raw(result["imageXref"]) == image.read_bytes()


def test_webp_alpha_fallback_ignores_only_its_pillow_size_limit(tmp_path, monkeypatch):
    source, image, output = tmp_path / "source.pdf", tmp_path / "image.webp", tmp_path / "output.pdf"
    _source(source)
    pixels = _quadrants(image, alpha=True)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    result = _place(source, image, output)
    assert Image.MAX_IMAGE_PIXELS == 10
    assert result["imagePixels"]["hasAlpha"]
    with pymupdf.open(output) as final:
        smask = int(final.xref_get_key(result["imageXref"], "SMask")[1].split()[0])
        mask = pymupdf.Pixmap(final, smask)
        assert np.frombuffer(mask.samples, dtype=np.uint8).reshape(40, 80).tolist() == pixels[..., 3].tolist()


@pytest.mark.parametrize("changes", [
    {"x_mm": -1}, {"y_mm": -1}, {"width_mm": 0}, {"height_mm": -1},
    {"x_mm": math.nan}, {"width_mm": math.inf}, {"height_mm": math.nan},
    {"page_number": 0}, {"page_number": 2}, {"page_number": 1.5},
    {"layer_name": "  "}, {"layer_name": None},
    {"width_mm": 300}, {"x_mm": 100, "width_mm": 40},
])
def test_invalid_placement_rejected_without_output(tmp_path, changes):
    source, image, output = tmp_path / "source.pdf", tmp_path / "image.png", tmp_path / "output.pdf"
    _source(source)
    _quadrants(image)
    with pytest.raises(PdfImageToolError):
        _place(source, image, output, **changes)
    assert not output.exists()


def test_bad_images_non_pdf_and_encryption_rejected(tmp_path):
    source, image, output = tmp_path / "source.pdf", tmp_path / "image.png", tmp_path / "output.pdf"
    _source(source)
    image.write_bytes(b"not an image")
    with pytest.raises(PdfImageToolError, match="图片"):
        _place(source, image, output)
    _quadrants(image)
    with pytest.raises(PdfImageToolError, match="不是 PDF"):
        _place(image, image, output)
    document = pymupdf.open(source)
    encrypted = tmp_path / "encrypted.pdf"
    document.save(encrypted, encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    document.close()
    with pytest.raises(PdfImageToolError, match="加密"):
        _place(encrypted, image, output)
    assert not output.exists()


def test_cancellation_and_input_output_collision(tmp_path):
    source, image, output = tmp_path / "source.pdf", tmp_path / "image.png", tmp_path / "output.pdf"
    _source(source)
    _quadrants(image)
    with pytest.raises(PdfProcessingCancelled):
        _place(source, image, output, cancelled=lambda: True)
    with pytest.raises(PdfImageToolError, match="覆盖"):
        _place(source, image, source)
    assert not output.exists()


def test_huge_page_uses_stream_copy_and_preserves_page_size(tmp_path, monkeypatch):
    source, image, output = tmp_path / "huge.pdf", tmp_path / "image.png", tmp_path / "output.pdf"
    document = pymupdf.open()
    document.new_page(width=100_000, height=70_000)
    document.save(source)
    document.close()
    _quadrants(image)
    original_read_bytes = Path.read_bytes

    def disallow_whole_pdf(self):
        if self.suffix.lower() == ".pdf":
            raise AssertionError("PDF must be copied through disk streams")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", disallow_whole_pdf)
    result = _place(source, image, output, max_image_pixels=80 * 40)
    assert result["pageSizeMm"]["width"] == pytest.approx(100_000 / MM_TO_PT)
    with pymupdf.open(output) as final:
        assert final[0].rect == pymupdf.Rect(0, 0, 100_000, 70_000)
        assert final.xref_get_key(result["imageXref"], "Width") == ("int", "80")


def test_previous_root_transform_and_clip_do_not_affect_new_image(tmp_path):
    source, image, output = tmp_path / "state.pdf", tmp_path / "image.png", tmp_path / "output.pdf"
    document = pymupdf.open()
    page = document.new_page(width=360, height=280)
    content = document.get_new_xref()
    document.update_object(content, "<<>>")
    original = b"0.8 0.8 0.8 rg 0 0 360 280 re f\n2 0 0 2 0 0 cm\n0 0 10 10 re W n\n"
    document.update_stream(content, original)
    page.set_contents(content)
    document.save(source)
    document.close()
    _quadrants(image)
    _place(source, image, output)
    with pymupdf.open(output) as final:
        assert final.xref_stream(content) == original
        pixmap = final[0].get_pixmap(colorspace=pymupdf.csRGB)
        pixels = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
        assert pixels[round(23 * MM_TO_PT), round(23 * MM_TO_PT)].tolist() == pytest.approx((255, 0, 0), abs=3)


def test_cancellation_during_stream_copy_leaves_no_result(tmp_path):
    source, image, output = tmp_path / "source.pdf", tmp_path / "image.png", tmp_path / "output.pdf"
    _source(source)
    _quadrants(image)
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls >= 3

    with pytest.raises(PdfProcessingCancelled):
        _place(source, image, output, cancelled=cancelled)
    assert not output.exists()
    assert not list(tmp_path.glob("pdf-image-*"))


def test_exact_page_edges_and_no_layer_source_are_supported(tmp_path):
    source, image, output = tmp_path / "blank.pdf", tmp_path / "image.png", tmp_path / "output.pdf"
    document = pymupdf.open()
    page = document.new_page(width=100 * MM_TO_PT, height=70 * MM_TO_PT)
    document.save(source)
    document.close()
    _quadrants(image)
    result = _place(source, image, output, x_mm=0, y_mm=0, width_mm=100, height_mm=70)
    with pymupdf.open(output) as final:
        assert len(final.get_ocgs()) == 1
        pixmap = final[0].get_pixmap(colorspace=pymupdf.csRGB)
        pixels = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
        assert pixels[2, 2].tolist() == pytest.approx((255, 0, 0), abs=3)
        assert pixels[-3, -3].tolist() == pytest.approx((255, 255, 0), abs=3)
    assert result["placementMm"]["width"] == 100


@pytest.mark.parametrize("extension", ["png", "jpg", "tif", "bmp", "webp"])
def test_upload_pixel_limit_checks_header_before_any_raster_decode(tmp_path, monkeypatch, extension):
    source, image, output = tmp_path / "source.pdf", tmp_path / f"image.{extension}", tmp_path / "output.pdf"
    _source(source)
    _quadrants(image)
    original_limit = Image.MAX_IMAGE_PIXELS

    def disallow_decode(*args, **kwargs):
        raise AssertionError("oversized image must be rejected before raster decoding")

    monkeypatch.setattr(pymupdf, "Pixmap", disallow_decode)
    monkeypatch.setattr(Image.Image, "load", disallow_decode)
    with pytest.raises(PdfImageToolError, match="超过上限"):
        _place(source, image, output, max_image_pixels=100)
    assert Image.MAX_IMAGE_PIXELS == original_limit
    assert not output.exists()


def test_header_pixel_budget_can_exceed_pillow_default_and_is_restored(tmp_path, monkeypatch):
    source, image, output = tmp_path / "source.pdf", tmp_path / "image.png", tmp_path / "output.pdf"
    _source(source)
    _quadrants(image)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    result = _place(source, image, output, max_image_pixels=3_200)
    assert Image.MAX_IMAGE_PIXELS == 10
    assert result["imagePixels"]["width"] == 80
