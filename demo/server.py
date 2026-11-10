"""Serve the demo and stream /api/* to a fixed upstream, using only stdlib."""
from __future__ import annotations

import argparse
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
from urllib.parse import parse_qs, unquote, urlsplit


CHUNK_BYTES = 64 * 1024
STATIC_FILES = {"/": ("index.html", "text/html; charset=utf-8"),
                "/index.html": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/styles.css": ("styles.css", "text/css; charset=utf-8"),
                "/vendor/pdf.mjs": ("vendor/pdf.mjs", "text/javascript; charset=utf-8"),
                "/vendor/pdf.worker.mjs": ("vendor/pdf.worker.mjs", "text/javascript; charset=utf-8"),
                "/vendor/LICENSE": ("vendor/LICENSE", "text/plain; charset=utf-8")}
HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
               "te", "trailer", "transfer-encoding", "upgrade", "proxy-connection"}
RESULT_PATH = re.compile(r"^/api/jobs/[^/]+/result$")


class InvalidRequest(ValueError):
    pass


def parse_api_base_url(value: str):
    parsed = urlsplit(value.strip())
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError("API 地址必须是固定的 http(s) URL，不支持账号、查询参数或 fragment")
    # Accessing .port validates malformed / out-of-range ports.
    parsed.port
    if any(char in value for char in "\r\n\\"):
        raise ValueError("API 地址包含无效字符")
    return parsed


def hop_headers(headers) -> set[str]:
    connection_tokens = {
        token.strip().lower()
        for value in headers.get_all("Connection", [])
        for token in value.split(",")
        if token.strip()
    }
    return HOP_HEADERS | connection_tokens


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, *, api_base_url: str, static_dir: Path | None = None,
                 upstream_timeout: float = 3600):
        self.upstream = parse_api_base_url(api_base_url)
        self.api_base_url = self.upstream.geturl().rstrip("/")
        self.static_dir = (static_dir or Path(__file__).resolve().parent / "dist").resolve()
        self.upstream_timeout = upstream_timeout
        super().__init__(address, DemoHandler)


class DemoHandler(BaseHTTPRequestHandler):
    server: DemoServer
    protocol_version = "HTTP/1.1"
    server_version = "ColorDemo"
    sys_version = ""

    def do_GET(self):
        self._dispatch()

    def do_HEAD(self):
        self._dispatch()

    def do_POST(self):
        self._dispatch()

    def do_DELETE(self):
        self._dispatch()

    def _json(self, status: int, detail):
        payload = json.dumps(detail, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        # Error requests may contain an unread upload body. Do not reinterpret
        # the remaining bytes as another request on a keep-alive connection.
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _dispatch(self):
        try:
            parsed = urlsplit(self.path)
            if parsed.scheme or parsed.netloc or "\\" in self.path:
                raise InvalidRequest("只接受本站相对路径")
            decoded = unquote(parsed.path, encoding="utf-8", errors="strict")
            if "\x00" in decoded:
                raise InvalidRequest("无效请求路径")
            if parsed.path.startswith("/api/"):
                if any(part in {".", ".."} for part in decoded.split("/")):
                    raise InvalidRequest("API 路径不能包含目录跳转")
                return self._proxy(parsed)
            if self.command not in {"GET", "HEAD"}:
                return self._json(405, {"detail": "此路径仅支持 GET / HEAD"})
            if decoded == "/demo-config":
                return self._json(200, {"apiBaseUrl": self.server.api_base_url})
            entry = STATIC_FILES.get(decoded)
            if entry is None:
                return self._json(404, {"detail": "资源不存在"})
            filename, content_type = entry
            path = (self.server.static_dir / filename).resolve()
            if not path.is_relative_to(self.server.static_dir) or not path.is_file():
                return self._json(404, {"detail": f"资源不存在：{filename}"})
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                with path.open("rb") as source:
                    while chunk := source.read(CHUNK_BYTES):
                        self.wfile.write(chunk)
        except (InvalidRequest, UnicodeError, ValueError) as exc:
            self._json(400, {"detail": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _request_framing(self) -> tuple[int | None, bool]:
        lengths = self.headers.get_all("Content-Length", [])
        encodings = self.headers.get_all("Transfer-Encoding", [])
        if len(lengths) > 1 or len(encodings) > 1 or (lengths and encodings):
            raise InvalidRequest("Content-Length / Transfer-Encoding 请求头冲突")
        if encodings:
            if encodings[0].strip().lower() != "chunked":
                raise InvalidRequest("仅支持 chunked Transfer-Encoding")
            return None, True
        if lengths:
            if not re.fullmatch(r"[0-9]+", lengths[0].strip()):
                raise InvalidRequest("Content-Length 必须是非负整数")
            return int(lengths[0]), False
        return 0, False

    def _read_exactly(self, size: int) -> bytes:
        data = self.rfile.read(size)
        if len(data) != size:
            raise InvalidRequest("上传尚未完成，客户端连接已断开")
        return data

    def _upload(self, upstream, length: int | None, chunked: bool):
        if not chunked:
            remaining = length or 0
            while remaining:
                data = self._read_exactly(min(CHUNK_BYTES, remaining))
                upstream.send(data)
                remaining -= len(data)
            return
        # Decode only HTTP transfer framing; multipart bytes remain untouched.
        while True:
            line = self.rfile.readline(8193)
            if len(line) > 8192 or not line.endswith(b"\r\n"):
                raise InvalidRequest("无效 chunked 请求头")
            size_text = line[:-2].split(b";", 1)[0]
            if not re.fullmatch(rb"[0-9a-fA-F]+", size_text):
                raise InvalidRequest("无效 chunked 块长度")
            remaining = int(size_text, 16)
            if not remaining:
                trailer_bytes = 0
                while True:
                    trailer = self.rfile.readline(8193)
                    trailer_bytes += len(trailer)
                    if not trailer.endswith(b"\r\n") or trailer_bytes > 65536:
                        raise InvalidRequest("无效 chunked trailer")
                    if trailer == b"\r\n":
                        break
                upstream.send(b"0\r\n\r\n")
                return
            while remaining:
                data = self._read_exactly(min(CHUNK_BYTES, remaining))
                upstream.send(f"{len(data):x}\r\n".encode("ascii"))
                upstream.send(data)
                upstream.send(b"\r\n")
                remaining -= len(data)
            if self._read_exactly(2) != b"\r\n":
                raise InvalidRequest("无效 chunked 块结尾")

    def _proxy(self, parsed):
        length, chunked = self._request_framing()
        target = self.server.upstream
        connection_type = http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection
        upstream = connection_type(target.hostname, target.port, timeout=self.server.upstream_timeout)
        response_started = False
        try:
            target_path = target.path.rstrip("/") + parsed.path
            if parsed.query:
                target_path += "?" + parsed.query
            upstream.putrequest(self.command, target_path, skip_accept_encoding=True)
            blocked = hop_headers(self.headers) | {"host", "expect", "content-length"}
            for name, value in self.headers.items():
                if name.lower() not in blocked:
                    upstream.putheader(name, value)
            if chunked:
                upstream.putheader("Transfer-Encoding", "chunked")
            elif length or self.command == "POST":
                upstream.putheader("Content-Length", str(length or 0))
            upstream.endheaders()
            try:
                self._upload(upstream, length, chunked)
            except (BrokenPipeError, ConnectionResetError):
                # Preserve an early upstream rejection, such as HTTP 413,
                # rather than disguising it as an unrelated proxy error.
                self.close_connection = True
            response = upstream.getresponse()
            preview = RESULT_PATH.fullmatch(parsed.path) and "1" in parse_qs(parsed.query).get("preview", [])
            blocked = hop_headers(response.headers) | {"server", "date"}
            known_length = response.headers.get("Content-Length")
            self.send_response(response.status, response.reason)
            response_started = True
            disposition_seen = False
            for name, value in response.getheaders():
                if name.lower() in blocked:
                    continue
                if preview and name.lower() == "content-disposition":
                    suffix = value.partition(";")[2]
                    value = "inline" + (";" + suffix if suffix else "")
                    disposition_seen = True
                self.send_header(name, value)
            if preview and not disposition_seen:
                self.send_header("Content-Disposition", "inline")
            if not known_length or self.close_connection:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            if self.command != "HEAD":
                while data := response.read(CHUNK_BYTES):
                    self.wfile.write(data)
                if response.length not in {None, 0}:
                    # A truncated upstream file must close the client socket;
                    # otherwise its advertised Content-Length would hang it.
                    self.close_connection = True
        except InvalidRequest:
            raise
        except (OSError, http.client.HTTPException) as exc:
            if response_started:
                self.close_connection = True
            else:
                self._json(502, {"detail": f"无法访问 API {self.server.api_base_url}：{exc}"})
        finally:
            upstream.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="运行颜色 API 的本地网页 demo 与流式转发")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5533)
    parser.add_argument("--api-base-url", default=os.getenv("API_BASE_URL", "http://localhost:5522"))
    args = parser.parse_args(argv)
    try:
        server = DemoServer((args.host, args.port), api_base_url=args.api_base_url)
    except ValueError as exc:
        parser.error(str(exc))
    with server:
        print(f"Demo: http://{args.host}:{server.server_port}/", flush=True)
        print(f"API:  {server.api_base_url}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
