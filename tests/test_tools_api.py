from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image
import pymupdf
import pytest
import zxingcpp

from api.config import Settings
from api.main import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(Settings(work_dir=tmp_path / "output", force_cpu=True))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def artwork():
    with pymupdf.open() as document:
        page = document.new_page(width=400, height=300)
        original_layer = document.add_ocg("Logo", on=True)
        technical = document.add_ocg("Technical", on=True)
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, 160, 80, bytes((80, 140, 170)) * (160 * 80), False)
        old_image = page.insert_image(pymupdf.Rect(10, 100, 210, 200), pixmap=pixmap, oc=original_layer)
        page.draw_line(pymupdf.Point(5, 10), pymupdf.Point(390, 10), color=(0, 0, 0), oc=technical)
        old_streams = {xref: document.xref_stream(xref) for xref in page.get_contents()}
        content = document.tobytes()
    buffer = BytesIO()
    Image.new("RGBA", (96, 48), (210, 40, 90, 128)).save(buffer, format="PNG")
    return {"pdf": content, "image": buffer.getvalue(), "old_image": old_image,
            "old_layer": original_layer, "technical": technical, "streams": old_streams}


def place(client, artwork, *, data=None, wait=True, image=None):
    fields = {"page": "1", "x_mm": "10", "y_mm": "10", "width_mm": "25.4", "height_mm": "12.7", "layer_name": "Logo"}
    if data:
        fields.update(data)
    return client.post(f"/api/tools/place-image?wait={'true' if wait else 'false'}", data=fields,
        files={"pdf": ("source.pdf", artwork["pdf"], "application/pdf"),
               "image": ("logo.png", image if image is not None else artwork["image"], "image/png")})


def terminal(client, job_id):
    for _ in range(200):
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200, response.text
        record = response.json()
        if record["status"] in {"succeeded", "failed", "cancelled"}:
            return record
        time.sleep(.02)
    pytest.fail("tool task did not finish")


def test_tools_are_in_one_new_swagger_tag_with_correct_output_media(client):
    spec = client.get("/openapi.json").json()
    for endpoint, media in (("place-image", "application/pdf"), ("qrcode", "image/png")):
        operation = spec["paths"][f"/api/tools/{endpoint}"]["post"]
        assert operation["tags"] == ["PDF 与二维码工具"]
        assert media in operation["responses"]["200"]["content"]
    fields = spec["paths"]["/api/tools/place-image"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]["properties"]
    assert {name for name, value in fields.items() if value.get("format") == "binary"} == {"pdf", "image"}
    assert fields["page"]["default"] == 1


def test_place_image_creates_independent_same_named_layer_and_retains_original_pixels(client, artwork):
    response = place(client, artwork)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/pdf"
    assert "-place-image.pdf" in response.headers["content-disposition"]
    assert "CMYK" not in response.headers["content-disposition"]
    with pymupdf.open(stream=response.content, filetype="pdf") as result:
        assert tuple(result[0].rect) == (0, 0, 400, 300)
        ocgs = result.get_ocgs()
        matches = [xref for xref, detail in ocgs.items() if detail["name"] == "Logo"]
        assert len(matches) == 2 and artwork["old_layer"] in matches
        assert ocgs[artwork["technical"]]["on"]
        for xref, stream in artwork["streams"].items():
            assert result.xref_stream(xref) == stream
        assert result.xref_get_key(artwork["old_image"], "Width") == ("int", "160")
        assert result.xref_get_key(artwork["old_image"], "Height") == ("int", "80")
        assert any(info["width"] == 96 and info["height"] == 48 for info in result[0].get_image_info())
    job_id = response.headers["x-job-id"]
    record = client.app.state.jobs.require(job_id)
    assert set(path.name for path in record.directory.iterdir()) == {"result.pdf", "result.json", "job.json"}
    assert client.get(f"/api/jobs/{job_id}/result").headers["content-type"] == "application/pdf"
    assert client.delete(f"/api/jobs/{job_id}").status_code == 200


def test_qrcode_png_is_transparent_and_decodes_exact_link(client):
    url = "https://example.com/颜色?token=A%2Fb&case=ABC#anchor"
    response = client.post("/api/tools/qrcode", json={"url": url})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    assert 'filename="qrcode.png"' in response.headers["content-disposition"]
    with Image.open(BytesIO(response.content)) as image:
        rgba = image.convert("RGBA")
        assert rgba.getpixel((0, 0))[3] == 0
        assert set(rgba.getchannel("A").getdata()) == {0, 255}
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        decoded = zxingcpp.read_barcode(background.convert("RGB"))
        assert decoded is not None and decoded.text == url
    record = client.app.state.jobs.require(response.headers["x-job-id"])
    assert set(path.name for path in record.directory.iterdir()) == {"result.png", "result.json", "job.json"}
    assert url not in (record.directory / "job.json").read_text(encoding="utf-8")
    assert url not in (record.directory / "result.json").read_text(encoding="utf-8")
    result = client.get(f"/api/jobs/{record.id}/result")
    assert result.status_code == 200 and result.headers["content-type"] == "image/png"
    assert result.content == response.content


def test_qrcode_can_then_be_placed_on_pdf(client, artwork):
    url = "https://example.com/label?id=42"
    qr = client.post("/api/tools/qrcode", json={"url": url})
    assert qr.status_code == 200, qr.text
    placed = place(client, artwork, data={"layer_name": "QR_Code", "width_mm": "30", "height_mm": "30"}, image=qr.content)
    assert placed.status_code == 200, placed.text
    with pymupdf.open(stream=placed.content, filetype="pdf") as document:
        scale = 144 / 72
        rectangle = pymupdf.Rect(10 / 25.4 * 72, 10 / 25.4 * 72, 40 / 25.4 * 72, 40 / 25.4 * 72)
        pixmap = document[0].get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=rectangle, alpha=False)
        image = Image.open(BytesIO(pixmap.tobytes("png")))
        decoded = zxingcpp.read_barcode(image)
        assert decoded is not None and decoded.text == url


def test_async_tool_result_is_downloadable(client, artwork):
    response = place(client, artwork, wait=False)
    assert response.status_code == 202, response.text
    accepted = response.json()
    assert terminal(client, accepted["id"])["status"] == "succeeded"
    result = client.get(accepted["result_url"])
    assert result.status_code == 200 and result.content.startswith(b"%PDF-")


def test_saved_png_is_downloadable_after_service_restart(tmp_path):
    settings = Settings(work_dir=tmp_path / "output", force_cpu=True)
    with TestClient(create_app(settings)) as first:
        response = first.post("/api/tools/qrcode", json={"url": "https://example.com/"})
        assert response.status_code == 200, response.text
        job_id, content = response.headers["x-job-id"], response.content
    with TestClient(create_app(settings)) as restarted:
        result = restarted.get(f"/api/jobs/{job_id}/result")
        assert result.status_code == 200 and result.headers["content-type"] == "image/png"
        assert result.content == content


@pytest.mark.parametrize("field,value", [("x_mm", "nan"), ("y_mm", "-1"), ("width_mm", "0"),
    ("height_mm", "inf"), ("page", "0"), ("page", "1.5"), ("layer_name", "   ")])
def test_bad_placement_fields_leave_no_job_or_slot_leak(client, artwork, field, value):
    response = place(client, artwork, data={field: value})
    assert response.status_code == 422, response.text
    assert client.app.state.jobs.list() == []
    assert client.post("/api/tools/qrcode", json={"url": "https://example.com/"}).status_code == 200


@pytest.mark.parametrize("url", ["not a link", "javascript:alert(1)", "https://example.com/a b", "https://example.com/%GG", "https://example.com:0/"])
def test_invalid_links_are_422(client, url):
    response = client.post("/api/tools/qrcode", json={"url": url})
    assert response.status_code == 422, response.text
    assert client.post("/api/tools/qrcode", json={"url": "https://example.com/"}).status_code == 200


def test_image_limit_and_pixel_budget_have_clear_errors(tmp_path, artwork):
    with TestClient(create_app(Settings(work_dir=tmp_path / "bytes", force_cpu=True,
                                        max_image_bytes=len(artwork["image"]) - 1))) as client:
        assert place(client, artwork).status_code == 413
        assert client.app.state.jobs.list() == []
    with TestClient(create_app(Settings(work_dir=tmp_path / "pixels", force_cpu=True,
                                        max_image_pixels=100))) as client:
        response = place(client, artwork)
        assert response.status_code == 422, response.text
        assert "像素" in response.text
        assert client.post("/api/tools/qrcode", json={"url": "https://example.com/"}).status_code == 200


def test_qrcode_body_size_is_bounded(client):
    response = client.post("/api/tools/qrcode", content=b"x" * 33000, headers={"content-type": "application/json"})
    assert response.status_code == 413


def test_tools_do_not_require_a_working_gpu(tmp_path, monkeypatch, artwork):
    import api.main as main
    monkeypatch.setattr(main, "detect_capabilities", lambda force_cpu: SimpleNamespace(gpu_available=False, note="GPU disabled for test"))
    with TestClient(create_app(Settings(work_dir=tmp_path / "output", force_cpu=False))) as client:
        assert client.get("/api/ready").status_code == 503
        assert client.post("/api/tools/qrcode", json={"url": "https://example.com/"}).status_code == 200
        assert place(client, artwork).status_code == 200


def test_cpu_api_finishes_while_gpu_worker_is_busy(client):
    jobs = client.app.state.jobs
    started, release = threading.Event(), threading.Event()
    record = jobs.reserve(original_filename="busy.pdf", options={}, has_pairs=False)
    def blocked(context):
        started.set()
        release.wait(10)
        context.record.result_pdf.write_bytes(b"%PDF-placeholder")
        return context.record.result_pdf
    future = jobs.submit(record.id, blocked)
    assert started.wait(2)
    try:
        before = time.monotonic()
        response = client.post("/api/tools/qrcode", json={"url": "https://example.com/"})
        assert response.status_code == 200, response.text
        assert time.monotonic() - before < 3
        assert not future.done()
    finally:
        release.set()
        future.result(timeout=5)


def test_qrcode_rejects_invalid_unicode_without_job_or_slot_leak(client):
    invalid_body = b'{"url":"https://example.com/\\ud800"}'
    response = client.post("/api/tools/qrcode", content=invalid_body, headers={"content-type": "application/json"})
    assert response.status_code == 422, response.text
    assert client.app.state.jobs.list() == []
    assert client.post("/api/tools/qrcode", json={"url": "https://example.com/"}).status_code == 200
