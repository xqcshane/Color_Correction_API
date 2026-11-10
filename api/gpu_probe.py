from __future__ import annotations

from typing import Any


def run_cupy_kernel_probe(cupy: Any) -> None:
    """Compile, execute and synchronize a tiny representative CUDA kernel.

    Device enumeration and memory allocation alone do not exercise NVRTC.  A
    broken CuPy include path can therefore look healthy until the first real
    color operation.  This probe intentionally uses ufuncs also used by the
    color engine so startup status reflects actual compute readiness.
    """

    values = cupy.arange(8, dtype=cupy.float32)
    result = cupy.where(values < 4, values * 2, values + 1)
    total = float(cupy.asnumpy(cupy.sum(result)))
    cupy.cuda.get_current_stream().synchronize()
    if abs(total - 38.0) > 1e-5:
        raise RuntimeError(f"CUDA 内核自检结果异常：{total}")


def concise_gpu_error(error: BaseException, limit: int = 600) -> str:
    text = " ".join(str(error).split()) or error.__class__.__name__
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return f"{error.__class__.__name__}: {text}"
