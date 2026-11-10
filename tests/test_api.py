from __future__ import annotations

import itertools
import json
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
import numpy as np
import pymupdf
import pytest

from api.config import Settings
from api.main import create_app


NEAREST_CSV = b"C,M,Y,K,L,a,b\n100,0,0,0,60,-20,-30\n"
MODEL_CSV = ("C,M,Y,K,L,a,b\n" + "".join(
    f"{c},{m},{y},{k},{100-.2*c-.15*m-.1*y-.4*k},{.4*m-.2*c},{.4*y-.1*m}\n"
    for c, m, y, k in itertools.product((0, 100), repeat=4)
)).encode()


@pytest.fixture
def artwork_pdf(tmp_path):
    path = tmp_path / "artwork.pdf"
    with pymupdf.open() as document:
        page = document.new_page(width=240, height=120)
        artwork = document.add_ocg("Artwork", on=True, usage="Artwork")
        technical = document.add_ocg("Technical", on=True, usage="Technical")
        rgb = np.zeros((60, 120, 3), dtype=np.uint8)
        rgb[..., :] = (220, 80, 0)
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, 120, 60, rgb.tobytes(), False)
        image = page.insert_image(page.rect, pixmap=pixmap, oc=artwork)
        page.draw_rect(pymupdf.Rect(5, 5, 235, 115), color=(0, 0, 0), width=2, oc=technical)
        technical_stream = document.xref_stream(page.get_contents()[-1])
        document.save(path)
    return {"bytes": path.read_bytes(), "artwork": artwork, "technical": technical,
            "image": image, "technical_stream": technical_stream}


@pytest.fixture
def client(tmp_path):
    app = create_app(Settings(work_dir=tmp_path / "runtime", force_cpu=True))
    with TestClient(app) as test_client:
        yield test_client


def post_pdf(client, pdf, *, endpoint="nearest", csv=NEAREST_CSV, layers='["Artwork"]', dpi=None, wait=True):
    data = {"layers": layers}
    if dpi is not None:
        data["dpi"] = str(dpi)
    return client.post(f"/api/{endpoint}?wait={'true' if wait else 'false'}", data=data,
                       files={"pdf": ("design.pdf", pdf, "application/pdf"), "csv": ("colors.csv", csv, "text/csv")})


def await_terminal(client, job_id):
    for _ in range(100):
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200, response.text
        info = response.json()
        if info["status"] in {"succeeded", "failed", "cancelled"}:
            return info
        time.sleep(.02)
    pytest.fail("job did not finish")


@pytest.mark.parametrize("endpoint,csv", [("nearest", NEAREST_CSV), ("predict", MODEL_CSV)])
def test_real_pdf_endpoints_return_fine_cmyk_preserving_original_density(client, artwork_pdf, endpoint, csv):
    response = post_pdf(client, artwork_pdf["bytes"], endpoint=endpoint, csv=csv)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF-")
    job = client.get(f"/api/jobs/{response.headers['x-job-id']}").json()
    assert job["status"] == "succeeded"
    assert job["result"]["info"]["quality"] == "fine"
    if endpoint == "nearest":
        assert job["result"]["info"]["rgbBits"] == 6
    else:
        assert job["result"]["info"]["lutSize"] == 13
        assert job["result"]["info"]["gridStep"] == 10
    with pymupdf.open(stream=response.content, filetype="pdf") as result:
        assert result.page_count == 1
        assert tuple(result[0].rect) == pytest.approx((0, 0, 240, 120))
        ocgs = result.get_ocgs()
        assert ocgs[artwork_pdf["technical"]]["on"] is True
        assert result.xref_object(artwork_pdf["technical"]).find("Technical") >= 0
        assert artwork_pdf["technical_stream"] in [result.xref_stream(ref) for ref in result[0].get_contents()]
        assert set(detail["name"] for detail in ocgs.values()) == {"Artwork", "Technical"}
        image = next(info for info in result[0].get_image_info(xrefs=True) if info["xref"] == artwork_pdf["image"])
        assert (image["width"], image["height"], image["colorspace"]) == (120, 60, 4)


def test_explicit_dpi_resamples_only_selected_layer(client, artwork_pdf):
    response = post_pdf(client, artwork_pdf["bytes"], dpi=72)
    assert response.status_code == 200, response.text
    with pymupdf.open(stream=response.content, filetype="pdf") as result:
        assert tuple(result[0].rect) == pytest.approx((0, 0, 240, 120))
        images = result[0].get_image_info()
        assert any(info["width"] == 240 and info["height"] == 120 and info["colorspace"] == 4 for info in images)
        assert result.get_ocgs()[artwork_pdf["technical"]]["on"] is True
        assert artwork_pdf["technical_stream"] in [result.xref_stream(ref) for ref in result[0].get_contents()]


def test_async_job_download_and_delete(client, artwork_pdf):
    response = post_pdf(client, artwork_pdf["bytes"], wait=False)
    assert response.status_code == 202, response.text
    accepted = response.json()
    assert accepted["status_url"] == f"/api/jobs/{accepted['id']}"
    info = await_terminal(client, accepted["id"])
    assert info["status"] == "succeeded", info
    result = client.get(accepted["result_url"])
    assert result.status_code == 200 and result.content.startswith(b"%PDF-")
    assert client.delete(accepted["status_url"]).status_code == 200
    assert client.get(accepted["status_url"]).status_code == 404


@pytest.mark.parametrize("layers,dpi", [("not-json", None), ("[]", None), ('["Missing"]', None),
                                       ('["Artwork","Artwork"]', None), ('[42]', None),
                                       ('["Artwork"]', "nan"), ('["Artwork"]', "0"),
                                       ('["Artwork"]', "abc")])
def test_bad_options_are_422_and_do_not_leave_jobs(client, artwork_pdf, layers, dpi):
    response = post_pdf(client, artwork_pdf["bytes"], layers=layers, dpi=dpi)
    assert response.status_code == 422, response.text
    assert list(client.app.state.settings.work_dir.joinpath("jobs").iterdir()) == []


def test_bad_pdf_and_csv_are_422(client, artwork_pdf):
    for pdf, csv in ((b"not a pdf", NEAREST_CSV), (artwork_pdf["bytes"], b"C,M,Y,K,L,a,b\nBAD,0,0,0,100,0,0\n")):
        response = post_pdf(client, pdf, csv=csv)
        assert response.status_code == 422, response.text
    assert list(client.app.state.settings.work_dir.joinpath("jobs").iterdir()) == []


@pytest.mark.parametrize("dpi", ["", " \t "])
def test_blank_dpi_preserves_original_native_density(client, artwork_pdf, dpi):
    response = post_pdf(client, artwork_pdf["bytes"], dpi=dpi)
    assert response.status_code == 200, response.text
    record = client.app.state.jobs.require(response.headers["x-job-id"])
    assert record.options["pdf"]["preserveNativeDensity"] is True
    assert record.options["pdf"]["dpi"] == pytest.approx(36)
    with pymupdf.open(stream=response.content, filetype="pdf") as document:
        native = next(info for info in document[0].get_image_info(xrefs=True) if info["xref"] == artwork_pdf["image"])
        assert (native["width"], native["height"], native["colorspace"]) == (120, 60, 4)


def test_swagger_docs_execute_interceptor_and_keep_two_file_contract(client):
    response = client.get("/docs/")
    assert response.status_code == 200
    assert response.history[0].status_code == 307
    html = response.text
    assert "requestInterceptor: function preserveMultipartJson(request)" in html
    schema = client.get("/openapi.json").json()
    for endpoint in ("/api/nearest", "/api/predict"):
        properties = schema["paths"][endpoint]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]["properties"]
        assert {key for key, value in properties.items() if value.get("format") == "binary"} == {"pdf", "csv"}
        assert properties["layers"]["type"] == "string"
        assert properties["layers"]["example"] == '["Artwork"]'
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to execute the Swagger bootstrap regression")
    inline = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(inline) == 1
    bootstrap_test = """
const assert = require('node:assert/strict');
global.location = {href: 'http://localhost/docs'};
let config, rawLayers, stateCalls = 0;
const SwaggerUIBundle = options => {
  config = options;
  return {getSystem: () => ({oas3Selectors: {requestBodyValue: (path, method) => {
    assert.ok(['/api/nearest', '/api/predict', '/api/tools/place-image'].includes(path));
    assert.equal(method, 'post');
    stateCalls++;
    return {getIn: keys => {assert.deepEqual(keys, [path === '/api/tools/place-image' ? 'layer_name' : 'layers', 'value']); return rawLayers;}};
  }}})};
};
SwaggerUIBundle.presets = {apis: {}};
SwaggerUIBundle.SwaggerUIStandalonePreset = {};
""" + inline[0] + """
assert.equal(typeof config.requestInterceptor, 'function');
assert.equal(config.showMutatedRequest, true);
for (const path of ['/api/nearest', '/api/predict']) {
  for (const raw of ['["Artwork"]', '["Artwork","Technical"]', '["Print,Artwork","中文 [图层]"]']) {
    rawLayers = raw;
    const body = new FormData();
    body.append('pdf', new Blob(['pdf']), 'source.pdf');
    body.append('csv', new Blob(['csv']), 'colors.csv');
    body.append('layers', JSON.parse(raw).join(','));
    body.append('dpi', '  ');
    const pdf = body.get('pdf'), csv = body.get('csv');
    const request = {url: 'http://localhost' + path, body};
    assert.equal(config.requestInterceptor(request), request);
    assert.equal(body.get('layers'), raw);
    assert.equal(body.has('dpi'), false);
    assert.equal(body.get('pdf'), pdf);
    assert.equal(body.get('csv'), csv);
    assert.equal([...body.values()].filter(value => typeof value !== 'string').length, 2);
    body.set('dpi', '72');
    config.requestInterceptor(request);
    assert.equal(body.get('dpi'), '72');
  }
}
for (const raw of ['QR_Code', '["QR","Label"]', '中文独立层']) {
  rawLayers = raw;
  const body = new FormData();
  body.append('pdf', new Blob(['pdf']), 'source.pdf');
  body.append('image', new Blob(['png']), 'image.png');
  body.append('layer_name', 'serialized-wrongly');
  config.requestInterceptor({url: 'http://localhost/api/tools/place-image', body});
  assert.equal(body.get('layer_name'), raw);
  assert.equal([...body.values()].filter(value => typeof value !== 'string').length, 2);
}
const callsBefore = stateCalls;
const otherBody = new FormData(); otherBody.append('layers', 'unchanged'); otherBody.append('dpi', '');
config.requestInterceptor({url: 'http://localhost/api/other', body: otherBody});
assert.equal(otherBody.get('layers'), 'unchanged'); assert.equal(otherBody.has('dpi'), true);
const jsonBody = {layers: 'unchanged'};
config.requestInterceptor({url: 'http://localhost/api/nearest', body: jsonBody});
config.requestInterceptor({url: 'http://localhost/openapi.json'});
assert.equal(jsonBody.layers, 'unchanged'); assert.equal(stateCalls, callsBefore);
"""
    result = subprocess.run([node], input=bootstrap_test, capture_output=True, text=True,
                            encoding="utf-8", timeout=15)
    assert result.returncode == 0, result.stderr


def test_validation_failure_releases_admission_slot(tmp_path, artwork_pdf):
    app = create_app(Settings(work_dir=tmp_path / "single-slot", force_cpu=True, max_pending_jobs=1))
    with TestClient(app) as client:
        assert post_pdf(client, artwork_pdf["bytes"], layers='["Missing"]').status_code == 422
        assert post_pdf(client, artwork_pdf["bytes"]).status_code == 200


def test_file_size_limit_and_body_limit(tmp_path, artwork_pdf):
    app = create_app(Settings(work_dir=tmp_path / "limits", force_cpu=True, max_pending_jobs=1,
                              max_upload_bytes=len(artwork_pdf["bytes"]) - 1, max_csv_bytes=1024))
    with TestClient(app) as client:
        for _ in range(2):
            assert post_pdf(client, artwork_pdf["bytes"]).status_code == 413
        response = client.post("/api/nearest", headers={"content-length": str(2 * 1024**2)})
        assert response.status_code == 413
        # A request after both failures reaches validation instead of a leaked
        # queue-slot response (429).
        assert client.post("/api/nearest").status_code == 422
    assert list(app.state.settings.work_dir.joinpath("jobs").iterdir()) == []


def test_chunked_body_limit_is_413(tmp_path):
    app = create_app(Settings(work_dir=tmp_path / "chunked", force_cpu=True,
                              max_upload_bytes=1, max_csv_bytes=1))
    boundary = b"test-boundary"
    head = b"--" + boundary + b'\r\nContent-Disposition: form-data; name="pdf"; filename="large.pdf"\r\nContent-Type: application/pdf\r\n\r\n'
    content = [head, b"x" * 1024**2, b"x" * 1024, b"\r\n--" + boundary + b"--\r\n"]
    with TestClient(app) as client:
        response = client.post("/api/nearest", content=iter(content),
                               headers={"content-type": "multipart/form-data; boundary=test-boundary"})
        assert response.status_code == 413, response.text
        assert client.post("/api/nearest").status_code == 422


def test_gpu_unavailable_is_503_before_upload(tmp_path, monkeypatch, artwork_pdf):
    import api.main
    capability = SimpleNamespace(gpu_available=False, gpu_name=None, note="CUDA kernel failed")
    monkeypatch.setattr(api.main, "detect_capabilities", lambda force_cpu: capability)
    app = create_app(Settings(work_dir=tmp_path / "no-gpu"))
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/ready").status_code == 503
        assert post_pdf(client, artwork_pdf["bytes"]).status_code == 503
    assert list(app.state.settings.work_dir.joinpath("jobs").iterdir()) == []


def test_queued_cancel_frees_slot_and_can_be_deleted(tmp_path, monkeypatch, artwork_pdf):
    import api.main
    release = threading.Event()
    started = threading.Event()

    def gated_processor(context):
        started.set()
        while not release.wait(.01):
            context.check_cancelled()
        shutil.copyfile(context.record.input_pdf, context.record.result_pdf)
        return context.record.result_pdf

    monkeypatch.setattr(api.main, "process_job", gated_processor)
    app = create_app(Settings(work_dir=tmp_path / "queue", force_cpu=True, max_pending_jobs=2))
    try:
        with TestClient(app) as client:
            first = post_pdf(client, artwork_pdf["bytes"], wait=False)
            assert first.status_code == 202
            assert started.wait(2)
            second = post_pdf(client, artwork_pdf["bytes"], wait=False)
            assert second.status_code == 202
            assert post_pdf(client, artwork_pdf["bytes"], wait=False).status_code == 429
            queued_id = second.json()["id"]
            assert client.post(f"/api/jobs/{queued_id}/cancel").json()["status"] == "cancelled"
            assert client.delete(f"/api/jobs/{queued_id}").status_code == 200
            third = post_pdf(client, artwork_pdf["bytes"], wait=False)
            assert third.status_code == 202, third.text
            release.set()
            assert await_terminal(client, first.json()["id"])["status"] == "succeeded"
            assert await_terminal(client, third.json()["id"])["status"] == "succeeded"
    finally:
        release.set()


def test_running_job_cancel_is_cooperative(tmp_path, monkeypatch, artwork_pdf):
    import api.main
    started = threading.Event()

    def cancellable(context):
        started.set()
        for _ in range(200):
            context.check_cancelled()
            time.sleep(.01)
        raise RuntimeError("test job should have been cancelled")

    monkeypatch.setattr(api.main, "process_job", cancellable)
    app = create_app(Settings(work_dir=tmp_path / "cancel", force_cpu=True, max_pending_jobs=1))
    with TestClient(app) as client:
        accepted = post_pdf(client, artwork_pdf["bytes"], wait=False)
        assert accepted.status_code == 202
        assert started.wait(2)
        job_id = accepted.json()["id"]
        assert client.delete(f"/api/jobs/{job_id}").status_code == 409
        assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 202
        assert await_terminal(client, job_id)["status"] == "cancelled"
        assert client.get(f"/api/jobs/{job_id}/result").status_code == 409
        assert client.delete(f"/api/jobs/{job_id}").status_code == 200


@pytest.mark.parametrize("wait", [True, False])
def test_disconnect_during_inspection_never_submits_and_releases_slot(tmp_path, monkeypatch, artwork_pdf, wait):
    import api.main
    from starlette.requests import Request

    inspected = threading.Event()
    disconnected_client = {"value": True}
    real_inspect = api.main.inspect_pdf
    real_processor = api.main.process_job

    def inspect_then_disconnect(path):
        metadata = real_inspect(path)
        inspected.set()
        return metadata

    async def disconnected(_request):
        # Simulate an abort after the body has been saved and its PDF checked,
        # matching the demo's wait=false upload/inspection phase.
        assert inspected.is_set()
        return disconnected_client["value"]

    processor = Mock(side_effect=AssertionError("disconnected upload reached GPU processing"))
    monkeypatch.setattr(api.main, "inspect_pdf", inspect_then_disconnect)
    monkeypatch.setattr(Request, "is_disconnected", disconnected)
    monkeypatch.setattr(api.main, "process_job", processor)
    app = create_app(Settings(work_dir=tmp_path / "disconnect", force_cpu=True, max_pending_jobs=1))
    submit = Mock(wraps=app.state.jobs.submit)
    monkeypatch.setattr(app.state.jobs, "submit", submit)
    with TestClient(app) as client:
        response = post_pdf(client, artwork_pdf["bytes"], wait=wait)
        assert response.status_code == 499, response.text
        assert inspected.is_set()
        submit.assert_not_called()
        processor.assert_not_called()
        assert app.state.jobs.list() == []
        assert list(app.state.settings.work_dir.joinpath("jobs").iterdir()) == []

        # A valid second request on the only admission slot must still enter
        # the queue and produce a PDF, rather than leaking a permanent 429.
        disconnected_client["value"] = False
        monkeypatch.setattr(api.main, "process_job", real_processor)
        accepted = post_pdf(client, artwork_pdf["bytes"], wait=False)
        assert accepted.status_code == 202, accepted.text
        submit.assert_called_once()
        finished = await_terminal(client, accepted.json()["id"])
        assert finished["status"] == "succeeded", finished


def test_matching_layer_name_selects_every_ocg_with_that_name(client, artwork_pdf):
    with pymupdf.open(stream=artwork_pdf["bytes"], filetype="pdf") as document:
        duplicate = document.add_ocg("Artwork", on=True, usage="Artwork")
        pixels = bytes((10, 90, 200)) * (60 * 60)
        image = pymupdf.Pixmap(pymupdf.csRGB, 60, 60, pixels, False)
        added_image = document[0].insert_image(pymupdf.Rect(60, 30, 120, 90), pixmap=image, oc=duplicate)
        source = document.tobytes()
    response = post_pdf(client, source)
    assert response.status_code == 200, response.text
    record = client.app.state.jobs.require(response.headers["x-job-id"])
    assert set(record.options["pdf"]["processedLayerIds"]) == {str(artwork_pdf["artwork"]), str(duplicate)}
    with pymupdf.open(stream=response.content, filetype="pdf") as output:
        for xref in (artwork_pdf["image"], added_image):
            assert output.xref_get_key(xref, "ColorSpace") == ("name", "/DeviceCMYK")
        assert all(output.get_ocgs()[xref]["on"] for xref in (artwork_pdf["artwork"], duplicate))
        assert artwork_pdf["technical_stream"] in [output.xref_stream(xref) for xref in output[0].get_contents()]


def test_swagger_tags_separate_both_algorithms_and_utility_endpoints(client):
    schema = client.get("/openapi.json").json()
    assert schema["paths"]["/api/nearest"]["post"]["tags"] == ["GPU 找最接近"]
    assert schema["paths"]["/api/predict"]["post"]["tags"] == ["GPU 推算新色"]
    assert schema["paths"]["/api/health"]["get"]["tags"] == ["服务状态"]
    assert schema["paths"]["/api/jobs/{job_id}"]["get"]["tags"] == ["任务管理"]
    assert {tag["name"] for tag in schema["tags"]} == {"GPU 找最接近", "GPU 推算新色", "PDF 与二维码工具", "服务状态", "任务管理"}
