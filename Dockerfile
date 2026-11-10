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

# Use authenticated Ubuntu mirrors and avoid stale proxy indexes during mirror sync.
# Keep by-hash optional for third-party repositories that do not support it.
RUN mkdir -p /data/tmp /data/cupy-cache \
    && find /etc/apt -maxdepth 2 -type f \( -name '*.list' -o -name '*.sources' \) \
      -exec sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' \
        -e 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' {} + \
    && printf '%s\n' \
        'Acquire::By-Hash "true";' \
        'Acquire::Retries "3";' \
        'Acquire::http::Timeout "20";' \
        'Acquire::https::Timeout "20";' \
        'Acquire::http::No-Cache "true";' \
        'Acquire::https::No-Cache "true";' \
        > /etc/apt/apt.conf.d/80-colorapi-network \
    && timeout 240s apt-get update \
    && timeout 300s apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
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
    && chown -R colorapi:colorapi /data

USER colorapi
EXPOSE 5522
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD ["python", "/app/scripts/healthcheck.py"]
CMD ["python", "-m", "api.run"]
