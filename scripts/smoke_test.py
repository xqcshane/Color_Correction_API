"""Exercise the packaged HTTP service with small synthetic PDFs, using CPU in CI.

Run inside a container started with CMYK_API_FORCE_CPU=true. This checks the
image, HTTP upload/download and real algorithms; GPU hardware needs a separate
check on the deployment host.
"""
from __future__ import annotations

import itertools
import json
import os
import time
import urllib.error
import urllib.request
import uuid

import pymupdf


BASE_URL = os.getenv("CMYK_SMOKE_BASE_URL", "http://127.0.0.1:5522").rstrip("/")


def request(path: str, *, data: bytes | None = None, headers: dict | None = None,
            method: str | None = None, timeout: float = 300):
    req = urllib.request.Request(BASE_URL + path, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"{req.method} {path}: HTTP {exc.code}: {detail}") from exc


def wait_ready() -> None:
    deadline = time.monotonic() + 90
    last_error = "service not started"
    while time.monotonic() < deadline:
        try:
            status, _, body = request("/api/ready", timeout=5)
            ready = json.loads(body)
            if status == 200 and ready.get("ok") and ready.get("backend") == "numpy":
                return
            last_error = repr(ready)
        except (OSError, RuntimeError, ValueError) as exc:
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f"Service did not become ready in explicit CPU test mode: {last_error}")


def make_pdf():
    with pymupdf.open() as document:
        page = document.new_page(width=240, height=120)
        artwork = document.add_ocg("Artwork", on=True)
        technical = document.add_ocg("Technical", on=True)
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, 120, 60, bytes((220, 80, 0)) * (120 * 60), False)
        page.insert_image(page.rect, pixmap=pixmap, oc=artwork)
        page.draw_rect(pymupdf.Rect(5, 5, 235, 115), color=(0, 0, 0), width=2, oc=technical)
        preserved_stream = document.xref_stream(page.get_contents()[-1])
        return document.tobytes(), preserved_stream


def form_data(files, fields):
    boundary = "colorapi-" + uuid.uuid4().hex
    parts = []
    for name, filename, content_type, content in files:
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                      f'filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n').encode())
        parts.extend((content, b"\r\n"))
    for name, value in fields.items():
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
                      f'\r\n\r\n{value}\r\n').encode())
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def multipart(pdf: bytes, csv: bytes, dpi: int | None):
    fields = {"layers": '["Artwork"]'}
    if dpi is not None:
        fields["dpi"] = str(dpi)
    return form_data((("pdf", "design.pdf", "application/pdf", pdf),
                      ("csv", "measurements.csv", "text/csv", csv)), fields)


def main() -> None:
    if os.getenv("CMYK_API_FORCE_CPU", "").lower() not in {"1", "true", "yes", "on"}:
        raise RuntimeError("Smoke test requires explicit CMYK_API_FORCE_CPU=true")
    wait_ready()
    pdf, preserved_stream = make_pdf()
    nearest_csv = b"C,M,Y,K,L,a,b\n100,0,0,0,60,-20,-30\n"
    model_csv = ("C,M,Y,K,L,a,b\n" + "".join(
        f"{c},{m},{y},{k},{100-.2*c-.15*m-.1*y-.4*k},{.4*m-.2*c},{.4*y-.1*m}\n"
        for c, m, y, k in itertools.product((0, 100), repeat=4)
    )).encode()
    for endpoint, csv, dpi, dimensions in (
        ("nearest", nearest_csv, None, (120, 60)),
        ("predict", model_csv, 72, (240, 120)),
    ):
        body, headers = multipart(pdf, csv, dpi)
        status, response_headers, content = request(f"/api/{endpoint}", data=body, headers=headers)
        assert status == 200, status
        assert response_headers.get_content_type() == "application/pdf", response_headers
        assert content.startswith(b"%PDF-"), content[:100]
        with pymupdf.open(stream=content, filetype="pdf") as result:
            assert result.page_count == 1
            assert tuple(result[0].rect) == (0, 0, 240, 120)
            assert {layer["name"] for layer in result.get_ocgs().values()} == {"Artwork", "Technical"}
            assert preserved_stream in [result.xref_stream(ref) for ref in result[0].get_contents()]
            assert any((image["width"], image["height"]) == dimensions and image["colorspace"] == 4
                       for image in result[0].get_image_info()), "Missing CMYK image at requested density"
        job_id = response_headers["X-Job-ID"]
        _, _, job_bytes = request(f"/api/jobs/{job_id}")
        job = json.loads(job_bytes)
        assert job["status"] == "succeeded", job
        assert job["result"]["info"]["quality"] == "fine", job
        request(f"/api/jobs/{job_id}", method="DELETE")
        print(f"PASS /api/{endpoint}: PDF with CMYK image {dimensions}, preserved Technical layer", flush=True)


    # Test the CPU utilities in the same packaged image, including persistent
    # PNG retrieval and placing that transparent image into a fresh OCG.
    from io import BytesIO
    from PIL import Image
    status, qr_headers, qr_png = request("/api/tools/qrcode",
        data=json.dumps({"url": "https://example.com/demo?color=cyan"}).encode(),
        headers={"Content-Type": "application/json"})
    assert status == 200 and qr_headers.get_content_type() == "image/png"
    with Image.open(BytesIO(qr_png)) as qr_image:
        rgba = qr_image.convert("RGBA")
        assert rgba.getpixel((0, 0))[3] == 0
        assert rgba.getchannel("A").getextrema() == (0, 255)
    qr_id = qr_headers["X-Job-ID"]
    _, saved_qr_headers, saved_qr = request(f"/api/jobs/{qr_id}/result")
    assert saved_qr_headers.get_content_type() == "image/png" and saved_qr == qr_png
    print("PASS /api/tools/qrcode: transparent PNG persisted and downloaded", flush=True)
    body, headers = form_data((("pdf", "design.pdf", "application/pdf", pdf),
                               ("image", "qrcode.png", "image/png", qr_png)),
        {"page": "1", "x_mm": "5", "y_mm": "5", "width_mm": "30", "height_mm": "30", "layer_name": "Artwork"})
    status, placed_headers, placed_pdf = request("/api/tools/place-image", data=body, headers=headers)
    assert status == 200 and placed_headers.get_content_type() == "application/pdf"
    with pymupdf.open(stream=placed_pdf, filetype="pdf") as result:
        assert tuple(result[0].rect) == (0, 0, 240, 120)
        assert sum(layer["name"] == "Artwork" for layer in result.get_ocgs().values()) == 2
        assert preserved_stream in [result.xref_stream(ref) for ref in result[0].get_contents()]
        assert any(image["width"] == 120 and image["height"] == 60 and image["colorspace"] == 3
                   for image in result[0].get_image_info()), "Original RGB pixels were changed"
    request(f"/api/jobs/{placed_headers['X-Job-ID']}", method="DELETE")
    request(f"/api/jobs/{qr_id}", method="DELETE")
    print("PASS /api/tools/place-image: fresh same-named layer, original pixels and Technical stream retained", flush=True)


if __name__ == "__main__":
    main()
