# =============================================================================
# OneTrainer-Ultimate — Linux x86_64 / Python 3.12 / CUDA 12.8 / PyTorch cu128
# RunPod-optimized with Web noVNC Desktop GUI, JupyterLab, FileBrowser,
# TensorBoard, FlashAttention-2, bitsandbytes, and persistent /workspace wiring.
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
    QT_QPA_PLATFORM=xcb \
    QT_X11_NO_MITSHM=1 \
    DISPLAY=:1 \
    VNC_RESOLUTION=1920x1080

# ---- 1. System Dependencies (Python 3.12 + X11/noVNC + Build Tools + SSH) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common gnupg ca-certificates curl \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
 && apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev python3-tk \
      git git-lfs aria2 wget \
      ffmpeg libsndfile1 libglib2.0-0 libgomp1 libgl1 \
      build-essential ninja-build \
      openssh-server \
      xvfb x11vnc openbox novnc websockify xterm \
      dbus-x11 libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
      libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-sync1 \
      libxcb-xfixes0 libxcb-xinerama0 libxcb-xkb1 libxkbcommon-x11-0 \
      libxkbcommon0 libfontconfig1 fonts-dejavu-core fonts-freefont-ttf \
      x11-xserver-utils x11-utils procps \
 && apt-get install -y --only-upgrade libstdc++6 \
 && { strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep -q GLIBCXX_3.4.32 \
      && echo "libstdc++ provides GLIBCXX_3.4.32 OK" \
      || { echo "FATAL: GLIBCXX_3.4.32 missing after libstdc++ upgrade"; exit 1; }; } \
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

# ---- 4. Attention Backends (FlashAttention-2 & SageAttention) ----
COPY scripts/ /opt/scripts/
RUN chmod +x /opt/scripts/*.sh /opt/scripts/*.py

RUN pip install --no-cache-dir \
      "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
   || pip install flash-attn==2.8.3.post1 --no-build-isolation \
   || echo "WARN: flash-attn install skipped or failed"

RUN bash -c '\
  try_wheel() { pip uninstall -y sageattention >/dev/null 2>&1 || true; \
                pip install --no-cache-dir --no-deps --force-reinstall "$1" \
                && python /opt/scripts/sage_check.py; }; \
  if try_wheel "https://huggingface.co/Kijai/PrecompiledWheels/resolve/main/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl"; then \
    echo "SageAttention: Kijai prebuilt wheel OK"; \
  elif try_wheel "https://github.com/thekie/sageattention-wheel/releases/download/2.2.0.post1/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl"; then \
    echo "SageAttention: thekie prebuilt wheel OK"; \
  else \
    echo "SageAttention: no usable prebuilt wheel -> compiling from source"; \
    pip uninstall -y sageattention || true; \
    ( git clone --depth 1 https://github.com/thu-ml/SageAttention.git /tmp/sage \
      && cd /tmp/sage \
      && TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;12.0" EXT_PARALLEL=2 NVCC_APPEND_FLAGS="--threads 4" MAX_JOBS=4 \
         pip install --no-build-isolation . ) \
      || echo "WARN: SageAttention source build failed; SDPA fallback active"; \
    rm -rf /tmp/sage; \
  fi'

# ---- 5. Pre-bake Core ML Ecosystem & Optimizers ----
RUN uv pip install --no-cache \
      numpy scipy matplotlib pillow \
      transformers tokenizers huggingface-hub hf_transfer accelerate safetensors peft \
      sentencepiece open-clip-torch gguf onnxruntime-gpu \
      bitsandbytes dadaptation lion-pytorch prodigyopt schedulefree \
      pytorch_optimizer prodigy-plus-schedule-free adv_optm \
      tensorboard jupyterlab nvitop psutil requests deepdiff \
      av scenedetect parse yt-dlp pooch imagesize \
      customtkinter PySide6

# Install Muon optimizer
RUN pip install --no-cache-dir git+https://github.com/KellerJordan/Muon.git@f90a42b

# ---- 6. OneTrainer Core Clone & Dependencies ----
RUN git clone https://github.com/Nerogar/OneTrainer.git /OneTrainer
WORKDIR /OneTrainer

# Install pinned git submodules / packages (diffusers & mgds)
RUN pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git@1ffa423 \
 && pip install --no-cache-dir git+https://github.com/Nerogar/mgds.git@3a6994a

# ---- 7. FileBrowser (pinned v2.31.2 binary) ----
RUN curl -fsSL https://github.com/filebrowser/filebrowser/releases/download/v2.31.2/linux-amd64-filebrowser.tar.gz -o /tmp/fb.tgz \
 && tar -xzf /tmp/fb.tgz -C /usr/local/bin filebrowser \
 && chmod +x /usr/local/bin/filebrowser && rm -f /tmp/fb.tgz \
 && /usr/local/bin/filebrowser version

# ---- 8. noVNC HTML Auto-connect Page Setup ----
RUN ln -sf /usr/share/novnc/vnc.html /usr/share/novnc/index.html \
 && sed -i 's/autoconnect: false/autoconnect: true/' /usr/share/novnc/app/ui.js 2>/dev/null || true

# ---- 9. Smoke Test Gate ----
RUN python /opt/scripts/smoke_test.py

# ---- 10. Ports & Entrypoint ----
EXPOSE 8080 8888 8081 6006 22 5900
WORKDIR /OneTrainer
ENTRYPOINT ["/opt/scripts/start.sh"]
