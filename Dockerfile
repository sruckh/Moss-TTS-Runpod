FROM runpod/base:1.0.3-cuda1281-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/huggingface-cache \
    HF_HUB_CACHE=/runpod-volume/huggingface-cache/hub \
    RUNPOD_INIT_TIMEOUT=2400

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    git git-lfs ca-certificates curl build-essential cmake ninja-build pkg-config ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && git lfs install

RUN pip install --no-cache-dir uv

COPY handler.py config.py serverless_engine.py /opt/moss-tts/
COPY bootstrap.sh /opt/bootstrap.sh

CMD ["bash", "/opt/bootstrap.sh"]
