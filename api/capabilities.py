from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

from .gpu_probe import concise_gpu_error, run_cupy_kernel_probe


@dataclass(frozen=True, slots=True)
class ComputeCapabilities:
    backend: str
    gpu_available: bool
    gpu_name: str | None
    gpu_memory_bytes: int | None
    cupy_available: bool
    cupy_version: str | None
    cuda_runtime_version: str | None
    cpu_count: int
    python_version: str
    platform: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _detect_with_cupy() -> tuple[
    bool,
    str | None,
    int | None,
    str | None,
    str | None,
    str | None,
]:
    try:
        import cupy  # type: ignore

        count = int(cupy.cuda.runtime.getDeviceCount())
        if count < 1:
            return False, None, None, str(cupy.__version__), None, "未检测到 CUDA 设备"
        device = cupy.cuda.Device(0)
        props = cupy.cuda.runtime.getDeviceProperties(device.id)
        raw_name = props.get("name")
        if isinstance(raw_name, bytes):
            gpu_name = raw_name.decode("utf-8", errors="replace")
        else:
            gpu_name = str(raw_name) if raw_name else None
        total_memory = int(props.get("totalGlobalMem", 0)) or None
        runtime_version = cupy.cuda.runtime.runtimeGetVersion()
        runtime_text = str(runtime_version) if runtime_version else None
        run_cupy_kernel_probe(cupy)
        return True, gpu_name, total_memory, str(cupy.__version__), runtime_text, None
    except Exception as error:
        version = None
        try:
            version = str(cupy.__version__)  # type: ignore[name-defined]
        except Exception:
            pass
        return False, None, None, version, None, concise_gpu_error(error)


def _detect_with_nvidia_smi() -> tuple[bool, str | None, int | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
        "--id=0",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
            shell=False,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return False, None, None
        first_line = completed.stdout.strip().splitlines()[0]
        name, memory_mib = (part.strip() for part in first_line.rsplit(",", 1))
        return True, name or None, int(float(memory_mib)) * 1024**2
    except (OSError, ValueError, subprocess.SubprocessError):
        return False, None, None


@lru_cache(maxsize=2)
def detect_capabilities(force_cpu: bool = False) -> ComputeCapabilities:
    if force_cpu:
        cupy_gpu = False
        cupy_version = None
        runtime = None
        cupy_error = None
        hardware_gpu, smi_name, smi_memory = _detect_with_nvidia_smi()
        name, memory = smi_name, smi_memory
    else:
        cupy_gpu, name, memory, cupy_version, runtime, cupy_error = _detect_with_cupy()
        hardware_gpu = cupy_gpu
        if not hardware_gpu:
            hardware_gpu, smi_name, smi_memory = _detect_with_nvidia_smi()
            name = name or smi_name
            memory = memory or smi_memory

    if force_cpu:
        backend = "cpu"
        note = "GPU 已被 CMYK_API_FORCE_CPU 禁用；计算将使用 CPU。"
    elif cupy_gpu:
        backend = "cuda"
        note = "CUDA/CuPy 内核自检通过；颜色计算使用 GPU（支持 RTX 3090）。"
    elif hardware_gpu:
        backend = "cpu"
        reason = f" 原因：{cupy_error}" if cupy_error else ""
        note = f"检测到 NVIDIA GPU，但 CuPy 实际内核自检失败；GPU 接口拒绝处理请求。{reason}"
    else:
        backend = "cpu"
        note = "未检测到可用 CUDA 运行环境；GPU 接口拒绝处理请求。"

    return ComputeCapabilities(
        backend=backend,
        gpu_available=cupy_gpu and not force_cpu,
        gpu_name=name,
        gpu_memory_bytes=memory,
        cupy_available=cupy_gpu,
        cupy_version=cupy_version,
        cuda_runtime_version=runtime,
        cpu_count=os.cpu_count() or 1,
        python_version=platform.python_version(),
        platform=platform.platform(),
        note=note,
    )
