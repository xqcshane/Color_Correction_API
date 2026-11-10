# CUDA devel includes NVRTC and CUDA headers required by CuPy kernel compilation.
# RTX 3090 is Ampere (compute capability 8.6); CuPy detects it at runtime.
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH \
    CUDA_PATH=/usr/local/cuda \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CMYK_API_WORK_DIR=/data \
    CUPY_CACHE_DIR=/data/cupy-cache \
    TMPDIR=/data/tmp

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

WORKDIR /app
COPY api/requirements*.txt /app/api/
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/api/requirements-gpu.txt \
    && python -m pip check

COPY api /app/api
COPY scripts /app/scripts
RUN groupadd --gid 10001 colorapi \
    && useradd --uid 10001 --gid colorapi --no-create-home colorapi \
    && mkdir -p /data/tmp /data/cupy-cache \
    && chown -R colorapi:colorapi /data

USER colorapi
EXPOSE 5522
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD ["python", "/app/scripts/healthcheck.py"]
CMD ["python", "-m", "api.run"]
