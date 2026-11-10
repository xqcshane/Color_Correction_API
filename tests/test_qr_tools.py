from __future__ import annotations

import socket
import urllib.request

import numpy as np
from PIL import Image
import pytest
import zxingcpp

from api.qr_tools import BORDER_MODULES, MAX_URL_BYTES, MODULE_SIZE, generate_qr_png


def decode_white_background(path):
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, "white")
    white.alpha_composite(rgba)
    symbols = zxingcpp.read_barcodes(white.convert("RGB"))
    assert len(symbols) == 1
    assert symbols[0].format == zxingcpp.BarcodeFormat.QRCode
    return symbols[0].text


@pytest.mark.parametrize("url", [
    "http://a",
    "https://example.com/",
    "HTTP://ExAmPle.COM:80/a/../b?x=1&x=2&encoded=%2f%2F#Frag",
    "https://例子.测试/颜色?布料=棉布&emoji=🎨#原稿",
    "http://localhost:5522/docs/",
    "https://[2001:db8::1]:5522/path?a=1&b=two%20words",
])
def test_generated_qr_really_decodes_to_exact_original_url(tmp_path, url):
    output = tmp_path / "result.png"
    metadata = generate_qr_png(url, output)
    assert decode_white_background(output) == url
    assert metadata["encodedBytes"] == len(url.encode("utf-8"))
    assert metadata["kind"] == "qr"
    assert metadata["mediaType"] == "image/png"
    assert metadata["errorCorrection"] == "M"
    assert isinstance(metadata["qrVersion"], int) and 1 <= metadata["qrVersion"] <= 40


def test_png_has_transparent_quiet_zone_opaque_black_modules_and_integer_grid(tmp_path):
    output = tmp_path / "result.png"
    metadata = generate_qr_png("https://example.com/print?id=12", output)
    with Image.open(output) as image:
        assert image.format == "PNG"
        pixels = np.asarray(image.convert("RGBA"))
    width = (metadata["matrixSize"] + BORDER_MODULES * 2) * MODULE_SIZE
    assert pixels.shape == (width, width, 4)
    assert (metadata["width"], metadata["height"]) == (width, width)
    assert metadata["transparentBackground"] is True
    alpha = pixels[..., 3]
    assert set(np.unique(alpha)) == {0, 255}
    border = BORDER_MODULES * MODULE_SIZE
    assert np.all(alpha[:border] == 0) and np.all(alpha[-border:] == 0)
    assert np.all(alpha[:, :border] == 0) and np.all(alpha[:, -border:] == 0)
    assert np.all(pixels[alpha == 255] == (0, 0, 0, 255))
    cells = pixels.reshape(width // MODULE_SIZE, MODULE_SIZE, width // MODULE_SIZE, MODULE_SIZE, 4)
    assert np.all(cells == cells[:, :1, :, :1, :])
    assert decode_white_background(output) == "https://example.com/print?id=12"


def test_long_near_capacity_url_decodes_without_truncation(tmp_path):
    prefix = "https://example.com/?q="
    url = prefix + "a" * (2300 - len(prefix))
    output = tmp_path / "result.png"
    metadata = generate_qr_png(url, output)
    assert metadata["encodedBytes"] == 2300
    assert metadata["qrVersion"] == 40
    assert decode_white_background(output) == url


@pytest.mark.parametrize("url", [
    None, 42, b"https://example.com", "", "example.com", "//example.com/path",
    "ftp://example.com/", "javascript:alert(1)", "https:///path", "http://",
    " https://example.com/", "https://example.com/ ", "https://example.com/a\nb",
    "https://example.com/\x00", "https://example.com/\x7f", "https://example.com/\u00a0",
    "https://example.com/%zz", "https://example.com/%", "https://example.com/\\path",
    'https://example.com/"quoted"', "https://bad|host/path", "https://-bad.example/path",
    "https://example..com/", "https://example.com:abc/", "http://example.com:65536/",
    "http://example.com:0/", "http://[not-ipv6]/", "https://example.com/\ud800",
])
def test_invalid_url_is_value_error_and_does_not_write_file(tmp_path, url):
    output = tmp_path / "result.png"
    with pytest.raises(ValueError):
        generate_qr_png(url, output)
    assert not output.exists()


def test_over_capacity_url_is_rejected_instead_of_truncated(tmp_path):
    output = tmp_path / "result.png"
    with pytest.raises(ValueError, match="过长"):
        generate_qr_png("https://example.com/" + "a" * MAX_URL_BYTES, output)
    assert not output.exists()


def test_utf8_byte_capacity_and_exact_encoder_capacity_are_both_checked(tmp_path):
    prefix = "https://example.com/?q="
    output = tmp_path / "result.png"
    # Short character counts can still exceed the encoded byte capacity.
    with pytest.raises(ValueError, match="过长"):
        generate_qr_png(prefix + "中" * 800, output)
    # The theoretical byte-mode capacity excludes the ECI header. Let the
    # encoder reject this boundary instead of silently dropping URL bytes.
    with pytest.raises(ValueError, match="过长"):
        generate_qr_png(prefix + "a" * (MAX_URL_BYTES - len(prefix)), output)
    assert not output.exists()


def test_generation_never_opens_network_or_uses_url_as_filename(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("QR generation attempted a network operation")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    output = tmp_path / "fixed-result.png"
    url = "https://example.com/path/../../other.png?token=A%2fB#fragment"
    generate_qr_png(url, output)
    assert [path.name for path in tmp_path.iterdir()] == ["fixed-result.png"]
    assert decode_white_background(output) == url
