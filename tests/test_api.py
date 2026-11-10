from __future__ import annotations

import itertools
import json
from pathlib import Path
import shutil
import threading
import time
from types import SimpleNamespace

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
                                       ('["Artwork"]', "nan"), ('["Artwork"]', "0")])
def test_bad_options_are_422_and_do_not_leave_jobs(client, artwork_pdf, layers, dpi):
    response = post_pdf(client, artwork_pdf["bytes"], layers=layers, dpi=dpi)
    assert response.status_code == 422, response.text
    assert list(client.app.state.settings.work_dir.joinpath("jobs").iterdir()) == []


def test_bad_pdf_and_csv_are_422(client, artwork_pdf):
    for pdf, csv in ((b"not a pdf", NEAREST_CSV), (artwork_pdf["bytes"], b"C,M,Y,K,L,a,b\nBAD,0,0,0,100,0,0\n")):
        response = post_pdf(client, pdf, csv=csv)
        assert response.status_code == 422, response.text
    assert list(client.app.state.settings.work_dir.joinpath("jobs").iterdir()) == []


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


def test_disconnected_sync_client_requests_job_cancellation(tmp_path, monkeypatch, artwork_pdf):
    import api.main
    from starlette.requests import Request

    async def disconnected(_request):
        return True

    def wait_for_cancellation(context):
        for _ in range(200):
            context.check_cancelled()
            time.sleep(.01)
        raise RuntimeError("disconnect did not cancel the task")

    monkeypatch.setattr(Request, "is_disconnected", disconnected)
    monkeypatch.setattr(api.main, "process_job", wait_for_cancellation)
    app = create_app(Settings(work_dir=tmp_path / "disconnect", force_cpu=True, max_pending_jobs=1))
    with TestClient(app) as client:
        response = post_pdf(client, artwork_pdf["bytes"])
        assert response.status_code == 499, response.text
        records = app.state.jobs.list()
        assert len(records) == 1
        assert await_terminal(client, records[0].id)["status"] == "cancelled"
