"""Docker readiness includes an executed CUDA kernel, not just a live HTTP port."""

import urllib.request


with urllib.request.urlopen("http://127.0.0.1:5522/api/ready", timeout=8) as response:
    if response.status != 200:
        raise SystemExit(1)
