from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pymupdf = pytest.importorskip("pymupdf")

from api.pdf_engine import (
    PdfEngineError,
    TileResult,
    choose_artwork_layer,
    inspect_pdf,
    plan_page_tiles,
    render_pdf_thumbnail,
    replace_pdf_layers,
)


def _create_layered_pdf(path: Path) -> tuple[int, int]:
    document = pymupdf.open()
    page = document.new_page(width=720, height=360)
    texture = document.add_ocg("Print_Artwork", on=True, usage="Artwork")
    cutline = document.add_ocg("Cut_Line", on=True, usage="Technical")
    page.draw_rect(
        pymupdf.Rect(0, 0, 720, 360),
        color=None,
        fill=(0.2, 0.6, 0.9),
        overlay=True,
        oc=texture,
    )
    page.draw_rect(
        pymupdf.Rect(20, 20, 700, 340),
        color=(1, 0, 0),
        width=4,
        overlay=True,
        oc=cutline,
    )
    document.save(path)
    document.close()
    return texture, cutline


def _create_image_layered_pdf(path: Path) -> tuple[int, int, int]:
    document = pymupdf.open()
    page = document.new_page(width=240, height=120)
    texture = document.add_ocg("Print_Artwork", on=True, usage="Artwork")
    cutline = document.add_ocg("Cut_Line", on=True, usage="Technical")
    rgb = np.zeros((60, 120, 3), dtype=np.uint8)
    rgb[..., 0] = 220
    rgb[..., 1] = 80
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, 120, 60, rgb.tobytes(), False)
    image_xref = page.insert_image(
        pymupdf.Rect(0, 0, 240, 120),
        pixmap=pixmap,
        keep_proportion=False,
        oc=texture,
    )
    page.draw_rect(
        pymupdf.Rect(5, 5, 235, 115),
        color=(0, 0, 0),
        width=2,
        oc=cutline,
    )
    document.save(path)
    document.close()
    return texture, cutline, image_xref


def _cyan_mapper(rgba: bytes, width: int, height: int, context) -> TileResult:
    pixels = np.frombuffer(rgba, dtype=np.uint8).reshape(height, width, 4)
    cmyk = np.zeros((height, width, 4), dtype=np.uint8)
    cmyk[..., 0] = 255
    return TileResult(
        cmyk=cmyk.tobytes(),
        stats={"visiblePixels": int(np.count_nonzero(pixels[..., 3]))},
    )


def test_plan_page_tiles_has_non_overlapping_cores() -> None:
    plans = plan_page_tiles(
        page_index=0,
        width_pt=720,
        height_pt=360,
        dpi=72,
        tile_size=256,
        bleed=2,
    )
    assert len(plans) == 6
    assert sum(plan.width * plan.height for plan in plans) == 720 * 360
    assert plans[0].core_in_outer == (0, 0, 256, 256)
    assert plans[1].core_in_outer[0] == 2


def test_inspect_and_thumbnail_layered_pdf(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    texture, _ = _create_layered_pdf(source)
    info = inspect_pdf(source)
    assert info["pageCount"] == 1
    assert info["pages"][0]["widthMm"] == pytest.approx(254.0)
    assert info["artworkLayerId"] == str(texture)
    assert [layer["name"] for layer in info["layers"]] == ["Print_Artwork", "Cut_Line"]
    thumbnail = render_pdf_thumbnail(source, max_dimension=320)
    assert thumbnail.startswith(b"\x89PNG")


def test_replace_only_selected_layer_preserves_page_and_other_layer(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    texture, cutline = _create_layered_pdf(source)
    result = replace_pdf_layers(
        source,
        output,
        selected_layer_xrefs=[texture],
        dpi=72,
        tile_size=256,
        mapper=_cyan_mapper,
    )
    assert output.is_file()
    assert result["unselectedObjectsPreserved"] is True
    assert result["pages"][0]["widthPt"] == pytest.approx(720)
    assert result["pages"][0]["heightPt"] == pytest.approx(360)
    with pymupdf.open(output) as document:
        ocgs = document.get_ocgs()
        assert ocgs[texture]["on"] is False
        assert ocgs[cutline]["on"] is True
        replacement = result["replacementLayers"][str(texture)]
        assert ocgs[replacement]["on"] is True
        assert all("布料与设备标注" != detail["name"] for detail in ocgs.values())
        assert any(image["colorspace"] == 4 for image in document[0].get_image_info())
        rendered = np.frombuffer(document[0].get_pixmap(colorspace=pymupdf.csRGB).samples, dtype=np.uint8)
        rendered = rendered.reshape(360, 720, 3)
        assert rendered[20, 20, 0] > rendered[20, 20, 1]


def test_native_image_replacement_keeps_xref_bbox_and_density(tmp_path: Path) -> None:
    source = tmp_path / "source-image.pdf"
    output = tmp_path / "output-image.pdf"
    texture, cutline, image_xref = _create_image_layered_pdf(source)
    with pymupdf.open(source) as before:
        source_xref_length = before.xref_length()
        image_before = next(item for item in before[0].get_image_info(xrefs=True) if item["xref"] == image_xref)
    info = inspect_pdf(source)
    layer = next(item for item in info["layers"] if item["id"] == str(texture))
    assert layer["directImageReplacement"] is True
    result = replace_pdf_layers(
        source,
        output,
        selected_layer_xrefs=[texture],
        dpi=36,
        tile_size=128,
        mapper=_cyan_mapper,
    )
    assert result["replacementMode"] == "native-images"
    assert result["replacedImageXrefs"][str(texture)] == [image_xref]
    with pymupdf.open(output) as after:
        assert after.xref_length() == source_xref_length
        assert after.get_ocgs()[texture]["on"] is True
        assert after.get_ocgs()[cutline]["on"] is True
        assert after.xref_get_key(image_xref, "ColorSpace") == ("name", "/DeviceCMYK")
        image_after = next(item for item in after[0].get_image_info(xrefs=True) if item["xref"] == image_xref)
        assert tuple(image_after["bbox"]) == pytest.approx(tuple(image_before["bbox"]))
        assert image_after["width"] == image_before["width"]
        assert image_after["height"] == image_before["height"]
        assert image_after["colorspace"] == 4


def test_explicit_density_really_resamples_native_artwork(tmp_path: Path) -> None:
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    texture, cutline, image_xref = _create_image_layered_pdf(source)
    with pymupdf.open(source) as before:
        original_streams = {xref: before.xref_stream(xref) for xref in before[0].get_contents()}
    result = replace_pdf_layers(
        source, output, selected_layer_xrefs=[texture], dpi=72,
        mapper=_cyan_mapper, preserve_native_density=False,
    )
    assert result["replacementMode"] == "native-images"
    with pymupdf.open(output) as document:
        assert document.get_ocgs()[cutline]["on"]
        assert document.get_ocgs()[texture]["on"]
        images = [item for item in document[0].get_image_info(xrefs=True) if item["xref"] == image_xref]
        assert images and max(item["width"] for item in images) == 240
        assert max(item["height"] for item in images) == 120
        for xref, content in original_streams.items():
            assert document.xref_stream(xref) == content


def test_unlayered_background_and_upper_guides_keep_stacking(tmp_path: Path) -> None:
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    document = pymupdf.open()
    page = document.new_page(width=200, height=100)
    artwork = document.add_ocg("Artwork")
    guides = document.add_ocg("Guides")
    page.draw_rect(page.rect, fill=(1, 1, 1), color=None)
    page.draw_rect(pymupdf.Rect(40, 20, 160, 80), fill=(1, 0, 0), color=None, oc=artwork)
    page.draw_line(pymupdf.Point(10, 50), pymupdf.Point(190, 50), color=(0, 0, 0), width=4, oc=guides)
    original_streams = {xref: document.xref_stream(xref) for xref in page.get_contents()}
    document.save(source)
    document.close()
    replace_pdf_layers(source, output, selected_layer_xrefs=[artwork], dpi=72, mapper=_cyan_mapper)
    with pymupdf.open(output) as result:
        # Original streams, including unlayered background and technical lines,
        # are retained byte-for-byte; only page stream ordering gets additions.
        for xref, data in original_streams.items():
            assert result.xref_stream(xref) == data
        pixels = np.frombuffer(result[0].get_pixmap(colorspace=pymupdf.csRGB).samples, dtype=np.uint8).reshape(100, 200, 3)
        assert np.all(pixels[10, 10] > 250)  # unrelated background is white
        assert pixels[30, 80, 0] < 100 and pixels[30, 80, 1] > 100  # artwork is cyan
        assert np.all(pixels[50, 80] < 20)  # upper guide remains above artwork


def test_nested_form_ocg_isolated_from_unlayered_content(tmp_path: Path) -> None:
    child = pymupdf.open()
    page = child.new_page(width=200, height=100)
    artwork = child.add_ocg("NestedArtwork")
    page.draw_rect(page.rect, color=None, fill=(1, 1, 1))
    page.draw_rect(pymupdf.Rect(40, 20, 160, 80), color=None, fill=(1, 0, 0), oc=artwork)
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    # One form containing both layer and unlayered paint is intentionally
    # rejected by the stacking check instead of producing reordered output.
    resources = child.xref_get_key(page.xref, "Resources")[1]
    data = b"\n".join(child.xref_stream(xref) for xref in page.get_contents())
    form_xref = child.get_new_xref()
    child.update_object(form_xref, f"<< /Type /XObject /Subtype /Form /BBox [0 0 200 100] /Resources {resources} >>")
    child.update_stream(form_xref, data)
    child.xref_set_key(page.xref, "Resources", f"<< /XObject << /Fm {form_xref} 0 R >> >>")
    content_xref = child.get_new_xref()
    child.update_object(content_xref, "<<>>")
    child.update_stream(content_xref, b"q /Fm Do Q")
    page.set_contents(content_xref)
    child.save(source)
    child.close()
    with pymupdf.open(source) as original:
        selected = next(xref for xref, detail in original.get_ocgs().items() if detail["name"] == "NestedArtwork")
    with pytest.raises(PdfEngineError, match="交错"):
        replace_pdf_layers(source, output, selected_layer_xrefs=[selected], dpi=72, mapper=_cyan_mapper)


def test_interleaved_layers_report_actionable_error(tmp_path: Path) -> None:
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    document = pymupdf.open()
    page = document.new_page(width=200, height=100)
    artwork, guides = document.add_ocg("Artwork"), document.add_ocg("Guides")
    page.draw_rect(pymupdf.Rect(0, 0, 100, 100), fill=(1, 0, 0), oc=artwork)
    page.draw_line(pymupdf.Point(0, 50), pymupdf.Point(200, 50), color=(0, 0, 0), oc=guides)
    page.draw_rect(pymupdf.Rect(100, 0, 200, 100), fill=(0, 1, 0), oc=artwork)
    document.save(source)
    document.close()
    with pytest.raises(PdfEngineError, match="保留原密度"):
        replace_pdf_layers(source, output, selected_layer_xrefs=[artwork], dpi=72, mapper=_cyan_mapper)


def test_oversized_image_uses_bounded_tiles(tmp_path: Path, monkeypatch) -> None:
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    texture, _, _ = _create_image_layered_pdf(source)
    monkeypatch.setenv("CMYK_PDF_NATIVE_MAX_PIXELS", "100")
    calls = []

    def mapper(rgba, width, height, context):
        calls.append((width, height))
        return _cyan_mapper(rgba, width, height, context)

    result = replace_pdf_layers(source, output, selected_layer_xrefs=[texture], dpi=72, tile_size=128, mapper=mapper)
    assert result["replacementMode"] == "ocg-overlay"
    assert len(calls) == 2
    assert all(width <= 132 and height <= 132 for width, height in calls)


def test_huge_page_thumbnail_obeys_pixel_limit(tmp_path: Path) -> None:
    source = tmp_path / "huge.pdf"
    document = pymupdf.open()
    document.new_page(width=100_000, height=60_000)
    document.save(source)
    document.close()
    pixmap = pymupdf.Pixmap(render_pdf_thumbnail(source, max_dimension=300))
    assert max(pixmap.width, pixmap.height) <= 301


def test_non_finite_density_rejected_before_planning() -> None:
    with pytest.raises(ValueError, match="dpi"):
        plan_page_tiles(page_index=0, width_pt=100, height_pt=100, dpi=float("inf"))


def test_selected_hidden_layer_is_processed(tmp_path: Path) -> None:
    source, output = tmp_path / "hidden.pdf", tmp_path / "output.pdf"
    document = pymupdf.open()
    page = document.new_page(width=200, height=100)
    artwork = document.add_ocg("HiddenArtwork", on=False)
    page.draw_rect(page.rect, color=None, fill=(1, 0, 0), oc=artwork)
    document.save(source)
    document.close()
    result = replace_pdf_layers(source, output, selected_layer_xrefs=[artwork], dpi=72, mapper=_cyan_mapper)
    with pymupdf.open(output) as document:
        pixels = np.frombuffer(document[0].get_pixmap(colorspace=pymupdf.csRGB).samples, dtype=np.uint8).reshape(100, 200, 3)
        assert pixels[30, 80, 0] < 100
        assert document.get_ocgs()[result["replacementLayers"][str(artwork)]]["on"]


def test_soft_alpha_remains_eight_bit_and_straight_cmyk(tmp_path: Path) -> None:
    source, output = tmp_path / "alpha.pdf", tmp_path / "output.pdf"
    document = pymupdf.open()
    page = document.new_page(width=200, height=100)
    artwork = document.add_ocg("Artwork")
    page.draw_rect(page.rect, color=None, fill=(1, 1, 1))
    page.draw_rect(pymupdf.Rect(40, 20, 160, 80), color=None, fill=(1, 0, 0), fill_opacity=0.5, oc=artwork)
    document.save(source)
    document.close()
    result = replace_pdf_layers(source, output, selected_layer_xrefs=[artwork], dpi=72, mapper=_cyan_mapper)
    with pymupdf.open(output) as document:
        replacement = result["replacementLayers"][str(artwork)]
        image = next(image for image in document[0].get_images(full=True) if document.get_oc(image[0]) == replacement)
        assert document.xref_get_key(image[0], "ColorSpace") == ("name", "/DeviceCMYK")
        assert document.xref_get_key(image[1], "BitsPerComponent") == ("int", "8")
        alpha = document.xref_stream(image[1])[30 * 200 + 80]
        assert 120 <= alpha <= 130
        assert document.xref_stream(image[0])[4 * (30 * 200 + 80):4 * (30 * 200 + 80) + 4] == bytes([255, 0, 0, 0])
        pixels = np.frombuffer(document[0].get_pixmap(colorspace=pymupdf.csRGB).samples, dtype=np.uint8).reshape(100, 200, 3)
        assert 110 < pixels[30, 80, 0] < 180


def test_single_layer_nested_form_keeps_transform(tmp_path: Path) -> None:
    source, output = tmp_path / "form.pdf", tmp_path / "output.pdf"
    document = pymupdf.open()
    page = document.new_page(width=200, height=100)
    artwork = document.add_ocg("NestedArtwork")
    page.draw_rect(pymupdf.Rect(40, 20, 160, 80), color=None, fill=(1, 0, 0), oc=artwork)
    resources = document.xref_get_key(page.xref, "Resources")[1]
    data = b"\n".join(document.xref_stream(xref) for xref in page.get_contents())
    form_xref = document.get_new_xref()
    document.update_object(form_xref, f"<< /Type /XObject /Subtype /Form /BBox [0 0 200 100] /Resources {resources} >>")
    document.update_stream(form_xref, data)
    document.xref_set_key(page.xref, "Resources", f"<< /XObject << /Fm {form_xref} 0 R >> >>")
    content_xref = document.get_new_xref()
    document.update_object(content_xref, "<<>>")
    document.update_stream(content_xref, b"q /Fm Do Q")
    page.set_contents(content_xref)
    document.save(source)
    document.close()
    replace_pdf_layers(source, output, selected_layer_xrefs=[artwork], dpi=72, mapper=_cyan_mapper)
    with pymupdf.open(output) as document:
        pixels = np.frombuffer(document[0].get_pixmap(colorspace=pymupdf.csRGB).samples, dtype=np.uint8).reshape(100, 200, 3)
        assert pixels[30, 80, 0] < 100
        assert np.all(pixels[10, 10] > 250)


def test_native_resample_clones_shared_soft_mask(tmp_path: Path) -> None:
    source, output = tmp_path / "shared-mask.pdf", tmp_path / "output.pdf"
    document = pymupdf.open()
    page = document.new_page(width=480, height=120)
    artwork, technical = document.add_ocg("Artwork"), document.add_ocg("Technical")
    rgb = np.zeros((60, 120, 3), dtype=np.uint8)
    rgb[..., 0] = 220
    alpha = np.broadcast_to(np.linspace(0, 255, 120, dtype=np.uint8), (60, 120))
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, 120, 60, rgb.tobytes(), False)
    rgba = pymupdf.Pixmap(pixmap, 1)
    rgba.set_alpha(np.ascontiguousarray(alpha).tobytes(), premultiply=1)
    selected_image = page.insert_image(pymupdf.Rect(0, 0, 240, 120), pixmap=rgba, oc=artwork)
    mask_xref = int(document.xref_get_key(selected_image, "SMask")[1].split()[0])
    rgb[..., 0] = 0
    rgb[..., 2] = 220
    other_image = page.insert_image(pymupdf.Rect(240, 0, 480, 120), pixmap=pymupdf.Pixmap(pymupdf.csRGB, 120, 60, rgb.tobytes(), False), oc=technical)
    document.xref_set_key(other_image, "SMask", f"{mask_xref} 0 R")
    document.save(source, deflate=True)
    document.close()
    with pymupdf.open(source) as document:
        original_mask = document.xref_stream(mask_xref)
        original_other = document.xref_stream(other_image)
        original_other_object = document.xref_object(other_image)
        original_mask_object = document.xref_object(mask_xref)
    result = replace_pdf_layers(source, output, selected_layer_xrefs=[artwork], dpi=72, mapper=_cyan_mapper, preserve_native_density=False)
    assert result["replacementMode"] == "native-images"
    with pymupdf.open(output) as document:
        new_mask = int(document.xref_get_key(selected_image, "SMask")[1].split()[0])
        assert new_mask != mask_xref
        assert document.xref_get_key(new_mask, "Width") == ("int", "240")
        assert document.xref_get_key(new_mask, "Height") == ("int", "120")
        assert document.xref_get_key(new_mask, "BitsPerComponent") == ("int", "8")
        assert document.xref_get_key(selected_image, "Width") == ("int", "240")
        assert document.xref_stream(mask_xref) == original_mask
        assert document.xref_stream(other_image) == original_other
        assert document.xref_object(other_image) == original_other_object
        assert document.xref_object(mask_xref) == original_mask_object
        scaled_alpha = np.frombuffer(document.xref_stream(new_mask), dtype=np.uint8).reshape(120, 240)
        assert 100 < scaled_alpha[60, 120] < 160


def test_native_resample_respects_rotated_image_axes(tmp_path: Path) -> None:
    source, output = tmp_path / "rotated.pdf", tmp_path / "output.pdf"
    document = pymupdf.open()
    page = document.new_page(width=120, height=240)
    artwork = document.add_ocg("Artwork")
    rgb = np.full((60, 120, 3), 100, dtype=np.uint8)
    image_xref = page.insert_image(page.rect, pixmap=pymupdf.Pixmap(pymupdf.csRGB, 120, 60, rgb.tobytes(), False), rotate=90, oc=artwork)
    before_bbox = page.get_image_info()[0]["bbox"]
    document.save(source)
    document.close()
    replace_pdf_layers(source, output, selected_layer_xrefs=[artwork], dpi=72, mapper=_cyan_mapper, preserve_native_density=False)
    with pymupdf.open(output) as document:
        assert document.xref_get_key(image_xref, "Width") == ("int", "240")
        assert document.xref_get_key(image_xref, "Height") == ("int", "120")
        assert document[0].get_image_info()[0]["bbox"] == pytest.approx(before_bbox)


def test_explicit_density_rejects_shared_image_at_different_scales(tmp_path: Path) -> None:
    source, output = tmp_path / "reused.pdf", tmp_path / "output.pdf"
    document = pymupdf.open()
    page = document.new_page(width=400, height=100)
    artwork = document.add_ocg("Artwork")
    rgb = np.full((60, 120, 3), 100, dtype=np.uint8)
    image_xref = page.insert_image(pymupdf.Rect(0, 0, 200, 100), pixmap=pymupdf.Pixmap(pymupdf.csRGB, 120, 60, rgb.tobytes(), False), oc=artwork)
    page.insert_image(pymupdf.Rect(200, 0, 300, 50), xref=image_xref)
    document.save(source)
    document.close()
    with pytest.raises(PdfEngineError, match="不同尺寸下复用"):
        replace_pdf_layers(source, output, selected_layer_xrefs=[artwork], dpi=72, mapper=_cyan_mapper, preserve_native_density=False)

