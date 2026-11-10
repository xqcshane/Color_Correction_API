"""Generate a standard transparent QR PNG without resolving or fetching its URL."""

from __future__ import annotations

import ipaddress
from pathlib import Path
import re
from urllib.parse import urlsplit

import segno


MODULE_SIZE = 10
BORDER_MODULES = 4
ERROR_CORRECTION = "M"
# Version 40-M has at most 2331 byte-mode payload bytes. The UTF-8 ECI header
# takes a little additional space; Segno checks the exact remaining capacity.
MAX_URL_BYTES = 2331
_BAD_PERCENT = re.compile(r"%(?![0-9A-Fa-f]{2})")
_HOST_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\Z")


def _validate_url(url: str) -> int:
    if not isinstance(url, str) or not url:
        raise ValueError("请提供完整的 HTTP 或 HTTPS 链接")
    if len(url) > MAX_URL_BYTES:
        raise ValueError("链接过长，超过单个标准二维码的容量；请缩短链接后重试")
    # urlsplit strips some leading controls and embedded tabs/newlines. Reject
    # them before parsing, and never use a reconstructed/normalized URL as data.
    if any(char.isspace() or ord(char) < 32 or 127 <= ord(char) <= 159 for char in url):
        raise ValueError("链接不能包含空白或控制字符；请使用有效的 URL 转义")
    if any(char in url for char in '\\<>"') or _BAD_PERCENT.search(url):
        raise ValueError("链接包含无效字符或无效的百分号转义")
    try:
        encoded_bytes = len(url.encode("utf-8"))
        parsed = urlsplit(url)
        host = parsed.hostname
        # Accessing .port validates nonnumeric or out-of-range explicit ports.
        port = parsed.port
    except (ValueError, UnicodeError) as exc:
        raise ValueError("链接的主机名、端口或字符编码无效") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or not host:
        raise ValueError("仅支持包含主机名的完整 HTTP 或 HTTPS 链接")
    if port is not None and port == 0:
        raise ValueError("链接端口必须在 1–65535 之间")
    try:
        if ":" in host:
            ipaddress.IPv6Address(host)
        else:
            ascii_host = host.encode("idna").decode("ascii").removesuffix(".")
            if len(ascii_host) > 253 or not ascii_host or any(not _HOST_LABEL.fullmatch(label) for label in ascii_host.split(".")):
                raise ValueError("invalid host")
    except (ValueError, UnicodeError) as exc:
        raise ValueError("链接主机名无效") from exc
    if encoded_bytes > MAX_URL_BYTES:
        raise ValueError("链接过长，超过单个标准二维码的容量；请缩短链接后重试")
    return encoded_bytes


def generate_qr_png(url: str, output_path: Path) -> dict:
    """Encode the exact original URL into a black/transparent, integer-grid PNG.

    The caller owns output_path and its parent directory. Only that path is
    written; the URL is never used as a filename and is never requested.
    """
    encoded_bytes = _validate_url(url)
    try:
        qr = segno.make_qr(url, error=ERROR_CORRECTION, mode="byte", encoding="utf-8",
                           eci=True, boost_error=False)
    except segno.DataOverflowError as exc:
        raise ValueError("链接过长，超过单个标准二维码的容量；请缩短链接后重试") from exc
    output_path = Path(output_path)
    # Explicit transparent RGBA selects indexed PNG alpha, avoiding grayscale
    # tRNS interpretation differences in readers while keeping modules opaque.
    qr.save(str(output_path), kind="png", scale=MODULE_SIZE, border=BORDER_MODULES,
            dark="#000000", light="#ffffff00")
    width, height = qr.symbol_size(scale=MODULE_SIZE, border=BORDER_MODULES)
    return {
        "kind": "qr",
        "mediaType": "image/png",
        "width": width,
        "height": height,
        "qrVersion": qr.version,
        "errorCorrection": qr.error,
        "moduleSize": MODULE_SIZE,
        "borderModules": BORDER_MODULES,
        "matrixSize": len(qr.matrix),
        "transparentBackground": True,
        "encodedBytes": encoded_bytes,
    }
