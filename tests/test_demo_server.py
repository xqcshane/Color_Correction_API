from __future__ import annotations

from contextlib import closing, contextmanager
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading

import pytest

from demo.server import CHUNK_BYTES, DemoServer, parse_api_base_url


@contextmanager
def running(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def exchange(server, method="GET", path="/", body=None, headers=None):
    with closing(http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)) as connection:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.headers, response.read()


class UpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def do_GET(self):
        self.respond()

    def do_DELETE(self):
        self.respond()

    def do_POST(self):
        self.respond()

    def read_body(self):
        total, digest, small_body = 0, hashlib.sha256(), bytearray()
        if self.headers.get("Transfer-Encoding") == "chunked":
            while True:
                size = int(self.rfile.readline().strip(), 16)
                if size == 0:
                    assert self.rfile.readline() == b"\r\n"
                    break
                chunk = self.rfile.read(size)
                assert self.rfile.read(2) == b"\r\n"
                total += len(chunk)
                digest.update(chunk)
                small_body.extend(chunk)
                self.server.first_chunk.set()
        else:
            remaining = int(self.headers.get("Content-Length", "0"))
            while remaining:
                chunk = self.rfile.read(min(CHUNK_BYTES, remaining))
                if not chunk:
                    break
                total += len(chunk)
                remaining -= len(chunk)
                digest.update(chunk)
                if total < 1024**2:
                    small_body.extend(chunk)
                self.server.first_chunk.set()
        return total, digest.hexdigest(), bytes(small_body)

    def respond(self):
        size, digest, body = self.read_body()
        self.server.received.append({"method": self.command, "path": self.path,
                                     "headers": self.headers, "body": body,
                                     "size": size, "sha256": digest})
        if self.path.startswith("/api/jobs/test/result"):
            data = b"%PDF-1.7 fake test PDF\n"
            code = 200
            if self.headers.get("Range") == "bytes=2-5":
                data, code = data[2:6], 206
            self.send_response(code)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Disposition", 'attachment; filename="result.pdf"')
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("ETag", '"test-pdf"')
            if code == 206:
                self.send_header("Content-Range", "bytes 2-5/23")
        elif self.path == "/api/retry":
            data = b'{"detail":"busy"}'
            self.send_response(429)
            self.send_header("Retry-After", "7")
            self.send_header("Content-Type", "application/json")
        elif self.path == "/api/unframed":
            data = b"body without content length"
            self.send_response(200)
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
            self.close_connection = True
            return
        elif self.path == "/api/download-stream":
            self.send_response(200)
            self.send_header("Content-Length", str(CHUNK_BYTES * 2))
            self.end_headers()
            self.wfile.write(b"a" * CHUNK_BYTES)
            self.wfile.flush()
            self.server.continue_download.wait(timeout=5)
            self.wfile.write(b"b" * CHUNK_BYTES)
            return
        elif self.path == "/api/truncated":
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(b"abc")
            self.close_connection = True
            return
        else:
            data = json.dumps({"size": size, "sha256": digest}).encode()
            self.send_response(202 if self.command == "DELETE" else 200)
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-End-To-End", "kept")
        self.send_header("Connection", "X-Upstream-Hop")
        self.send_header("X-Upstream-Hop", "removed")
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def servers(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    for name, content in {"index.html": "<h1>demo</h1>", "app.js": "console.log('demo')",
                          "styles.css": "body { color: black; }"}.items():
        (dist / name).write_text(content, encoding="utf-8")
    (dist / "vendor").mkdir()
    for name, content in {"pdf.mjs": "export const version = 'test';",
                          "pdf.worker.mjs": "export const worker = true;",
                          "LICENSE": "PDF.js license test fixture"}.items():
        (dist / "vendor" / name).write_text(content, encoding="utf-8")
    (tmp_path / "secret.txt").write_text("private", encoding="utf-8")
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream.daemon_threads = True
    upstream.received = []
    upstream.first_chunk = threading.Event()
    upstream.continue_download = threading.Event()
    with running(upstream):
        demo = DemoServer(("127.0.0.1", 0),
                          api_base_url=f"http://127.0.0.1:{upstream.server_port}", static_dir=dist)
        with running(demo):
            yield demo, upstream


def test_demo_config_and_static_whitelist(servers):
    demo, upstream = servers
    status, headers, data = exchange(demo)
    assert status == 200 and data == b"<h1>demo</h1>"
    assert headers["Content-Type"].startswith("text/html")
    assert int(headers["Content-Length"]) == len(data)
    for filename in ("app.js", "styles.css", "index.html"):
        assert exchange(demo, path="/" + filename)[0] == 200
    assert json.loads(exchange(demo, path="/demo-config")[2]) == {
        "apiBaseUrl": f"http://127.0.0.1:{upstream.server_port}"}
    assert exchange(demo, "HEAD", "/app.js")[2] == b""
    assert exchange(demo, "POST", "/app.js")[0] == 405
    assert not upstream.received


@pytest.mark.parametrize("filename,content_type", [
    ("pdf.mjs", "text/javascript"), ("pdf.worker.mjs", "text/javascript"), ("LICENSE", "text/plain")])
def test_pdfjs_whitelisted_vendor_assets(servers, filename, content_type):
    status, headers, body = exchange(servers[0], path="/vendor/" + filename)
    assert status == 200 and body
    assert headers["Content-Type"].startswith(content_type)
    assert int(headers["Content-Length"]) == len(body)


def test_whitelisted_vendor_symlink_cannot_escape_static_root(servers):
    demo, _ = servers
    path = demo.static_dir / "vendor" / "pdf.mjs"
    path.unlink()
    try:
        path.symlink_to(demo.static_dir.parent / "secret.txt")
    except OSError as exc:
        pytest.skip(f"Symlinks unavailable on this host: {exc}")
    assert exchange(demo, path="/vendor/pdf.mjs")[0] == 404


@pytest.mark.parametrize("path", ["/../secret.txt", "/%2e%2e/secret.txt", "/demo/server.py",
                                   "/dist/index.html", "/app.js/../index.html", "/missing.css",
                                   "/vendor/unknown.mjs", "/vendor/../app.js",
                                   "/vendor/%2e%2e/secret.txt", "/vendor/pdf.mjs/../../index.html"])
def test_static_cannot_serve_outside_whitelist(servers, path):
    assert exchange(servers[0], path=path)[0] == 404


def test_multipart_upload_preserves_bytes_headers_and_fixed_host(servers):
    demo, upstream = servers
    body = (b'--boundary\r\nContent-Disposition: form-data; name="pdf"; filename="design.pdf"\r\n'
            b'Content-Type: application/pdf\r\n\r\n%PDF-\x00\xff\n\r\n--boundary--\r\n')
    headers = {"Content-Type": "multipart/form-data; boundary=boundary", "Host": "untrusted.example",
               "Connection": "X-Client-Hop", "X-Client-Hop": "removed", "X-Pass": "yes"}
    status, response_headers, result = exchange(demo, "POST", "/api/nearest?wait=false", body, headers)
    assert status == 200 and json.loads(result)["size"] == len(body)
    received = upstream.received[0]
    assert received["body"] == body
    assert received["headers"]["Content-Type"] == headers["Content-Type"]
    assert received["headers"]["Host"] == f"127.0.0.1:{upstream.server_port}"
    assert received["headers"]["X-Pass"] == "yes"
    assert received["headers"].get("X-Client-Hop") is None
    assert response_headers["X-End-To-End"] == "kept"
    assert response_headers.get("X-Upstream-Hop") is None
    assert received["path"] == "/api/nearest?wait=false"


def test_large_upload_reaches_upstream_before_upload_finishes(servers):
    demo, upstream = servers
    chunk = b"p" * CHUNK_BYTES
    chunk_count = 128
    with closing(http.client.HTTPConnection("127.0.0.1", demo.server_port, timeout=5)) as connection:
        connection.putrequest("POST", "/api/large")
        connection.putheader("Content-Length", str(len(chunk) * chunk_count))
        connection.endheaders()
        connection.send(chunk)
        # A proxy that reads the entire upload before forwarding cannot pass.
        assert upstream.first_chunk.wait(timeout=3), "proxy buffered the complete request body"
        for _ in range(chunk_count - 1):
            connection.send(chunk)
        response = connection.getresponse()
        result = json.loads(response.read())
    expected = hashlib.sha256()
    for _ in range(chunk_count):
        expected.update(chunk)
    assert result == {"size": CHUNK_BYTES * chunk_count, "sha256": expected.hexdigest()}


def test_chunked_upload_is_streamed_without_changing_payload(servers):
    demo, upstream = servers
    chunks = [b"first\x00", b"second\xff", b"last"]
    with closing(http.client.HTTPConnection("127.0.0.1", demo.server_port, timeout=5)) as connection:
        connection.request("POST", "/api/nearest", body=iter(chunks), encode_chunked=True)
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["size"] == len(b"".join(chunks))
    assert upstream.received[0]["body"] == b"".join(chunks)


def test_range_pdf_preview_preserves_download_headers(servers):
    demo, upstream = servers
    status, headers, body = exchange(demo, path="/api/jobs/test/result?preview=1",
                                     headers={"Range": "bytes=2-5", "If-Range": '"test-pdf"'})
    assert status == 206 and body == b"DF-1"
    assert headers["Content-Disposition"] == 'inline; filename="result.pdf"'
    assert headers["Content-Type"] == "application/pdf"
    assert headers["Content-Range"] == "bytes 2-5/23"
    assert headers["Content-Length"] == "4"
    assert headers["Accept-Ranges"] == "bytes"
    assert headers["ETag"] == '"test-pdf"'
    assert upstream.received[0]["headers"]["Range"] == "bytes=2-5"
    assert upstream.received[0]["headers"]["If-Range"] == '"test-pdf"'
    assert exchange(demo, path="/api/jobs/test/result")[1]["Content-Disposition"].startswith("attachment")


def test_upstream_error_status_retry_after_and_delete_are_preserved(servers):
    demo, upstream = servers
    status, headers, body = exchange(demo, path="/api/retry")
    assert status == 429 and headers["Retry-After"] == "7"
    assert json.loads(body) == {"detail": "busy"}
    assert exchange(demo, "DELETE", "/api/jobs/test")[0] == 202
    assert upstream.received[-1]["method"] == "DELETE"


def test_unframed_upstream_download_terminates_cleanly(servers):
    status, headers, body = exchange(servers[0], path="/api/unframed")
    assert status == 200 and body == b"body without content length"
    assert headers["Connection"] == "close"


def test_download_reaches_browser_before_upstream_finishes(servers):
    demo, upstream = servers
    with closing(http.client.HTTPConnection("127.0.0.1", demo.server_port, timeout=3)) as connection:
        connection.request("GET", "/api/download-stream")
        response = connection.getresponse()
        try:
            assert response.read(CHUNK_BYTES) == b"a" * CHUNK_BYTES
        finally:
            upstream.continue_download.set()
        assert response.read() == b"b" * CHUNK_BYTES


def test_truncated_upstream_download_closes_instead_of_hanging(servers):
    with pytest.raises(http.client.IncompleteRead):
        exchange(servers[0], path="/api/truncated")


@pytest.mark.parametrize("path", ["http://untrusted.example/api/health", "/api/../demo-config"])
def test_request_cannot_select_another_api_target(servers, path):
    demo, upstream = servers
    assert exchange(demo, path=path)[0] == 400
    assert not upstream.received


def test_conflicting_request_framing_is_rejected(servers):
    demo, upstream = servers
    with closing(http.client.HTTPConnection("127.0.0.1", demo.server_port, timeout=5)) as connection:
        connection.putrequest("POST", "/api/nearest")
        connection.putheader("Content-Length", "4")
        connection.putheader("Transfer-Encoding", "chunked")
        connection.endheaders()
        response = connection.getresponse()
        assert response.status == 400
        assert "冲突" in json.loads(response.read())["detail"]
    assert not upstream.received


def test_unavailable_upstream_returns_readable_json(tmp_path):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    demo = DemoServer(("127.0.0.1", 0), api_base_url=f"http://127.0.0.1:{port}",
                      static_dir=tmp_path, upstream_timeout=.2)
    with running(demo):
        status, headers, body = exchange(demo, path="/api/health")
    assert status == 502 and headers["Content-Type"].startswith("application/json")
    assert "无法访问 API" in json.loads(body)["detail"]


@pytest.mark.parametrize("url", ["ftp://localhost:5522", "http://user:password@localhost",
                                 "http://localhost?target=evil", "http://localhost/#fragment",
                                 "http://localhost:99999", "http://localhost\\evil"])
def test_api_target_must_be_a_fixed_valid_http_url(url):
    with pytest.raises(ValueError):
        parse_api_base_url(url)
