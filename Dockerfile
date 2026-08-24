# =============================================================================
# OneTrainer-Ultimate — Linux x86_64 / Python 3.12 / CUDA 12.8 / PyTorch cu128
# Fast build with pre-compiled wheels (FlashAttention-2, bitsandbytes, Triton)
# =============================================================================
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin \
    PIP_CONSTRAINT=/opt/constraints.txt \
    UV_CONSTRAINT=/opt/constraints.txt \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=8080

# ---- 1. System Dependencies (Python 3.12 + Build Tools + SSH) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common gnupg ca-certificates curl \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev \
      git git-lfs aria2 wget \
      ffmpeg libsndfile1 libglib2.0-0 libgomp1 libgl1 \
      build-essential ninja-build \
      openssh-server procps \
 && git lfs install \
 && ssh-keygen -A \
 && sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
 && sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config \
 && rm -rf /var/lib/apt/lists/*

# ---- 2. Python 3.12 venv + Constraints ----
COPY constraints.txt /opt/constraints.txt
RUN python3.12 -m venv /opt/venv \
 && python -m pip install --upgrade pip setuptools wheel uv

# ---- 3. PyTorch cu128 ----
RUN pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 \
      --index-url https://download.pytorch.org/whl/cu128 \
 && python -c "import torch; assert torch.version.cuda=='12.8', torch.version.cuda; assert '+cu128' in torch.__version__, torch.__version__; print('PyTorch OK:', torch.__version__)"

# ---- 4. Attention Backends (Pre-built FlashAttention-2 & SageAttention Wheels) ----
COPY scripts/ /opt/scripts/
RUN chmod +x /opt/scripts/*.sh /opt/scripts/*.py

# Pre-compiled FlashAttention-2 wheel (matches torch 2.8.0+cu128 / python 3.12)
RUN pip install --no-cache-dir \
      "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
   || pip install flash-attn==2.8.3.post1 --no-build-isolation \
   || echo "WARN: flash-attn install skipped (SDPA fallback active)"

# Pre-compiled SageAttention wheel (optional inference acceleration)
RUN pip install --no-cache-dir --no-deps \
      "https://huggingface.co/Kijai/PrecompiledWheels/resolve/main/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl" \
   || pip install --no-cache-dir --no-deps \
      "https://github.com/thekie/sageattention-wheel/releases/download/2.2.0.post1/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl" \
   || echo "WARN: SageAttention wheel skipped (SDPA fallback active)"

# ---- 5. Pre-bake Core ML Ecosystem, Web Framework & Optimizers ----
RUN uv pip install --no-cache \
      fastapi uvicorn websockets \
      numpy scipy matplotlib pillow opencv-contrib-python-headless \
      transformers tokenizers huggingface-hub hf_transfer accelerate safetensors \
      sentencepiece open-clip-torch gguf onnxruntime-gpu \
      bitsandbytes dadaptation lion-pytorch prodigyopt schedulefree \
      pytorch_optimizer prodigy-plus-schedule-free adv_optm \
      tensorboard jupyterlab nvitop psutil requests deepdiff \
      av scenedetect parse yt-dlp pooch imagesize

# Install Muon optimizer
RUN pip install --no-cache-dir git+https://github.com/KellerJordan/Muon.git@f90a42b

# ---- 6. OneTrainer Core Clone & Dependencies ----
RUN git clone https://github.com/Nerogar/OneTrainer.git /OneTrainer
WORKDIR /OneTrainer

# Install pinned git submodules / packages (diffusers & mgds)
RUN pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git@1ffa423 \
 && pip install --no-cache-dir git+https://github.com/Nerogar/mgds.git@3a6994a

# Install WebUI Package into /OneTrainer/webui
COPY webui /OneTrainer/webui
COPY webui /opt/webui

# ---- 7. FileBrowser (pinned v2.31.2 binary) ----
RUN curl -fsSL https://github.com/filebrowser/filebrowser/releases/download/v2.31.2/linux-amd64-filebrowser.tar.gz -o /tmp/fb.tgz \
 && tar -xzf /tmp/fb.tgz -C /usr/local/bin filebrowser \
 && chmod +x /usr/local/bin/filebrowser && rm -f /tmp/fb.tgz \
 && /usr/local/bin/filebrowser version

# ---- 8. Smoke Test Gate ----
RUN python -c "from modules.util.config.TrainConfig import TrainConfig; print('OneTrainer core OK'); from webui.server import app, parse_train_config; print('WebUI App OK')"

# ---- 9. Ports & Entrypoint ----
EXPOSE 8080 8888 8081 6006 22
WORKDIR /OneTrainer
ENTRYPOINT ["/opt/scripts/start.sh"]
