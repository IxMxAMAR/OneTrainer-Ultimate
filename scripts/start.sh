#!/usr/bin/env bash
# =============================================================================
# RunPod Pod Entrypoint: OneTrainer Ultimate
# Starts SSH, JupyterLab, FileBrowser, TensorBoard, and the native OneTrainer WebUI.
# =============================================================================
set -u
export PATH="/opt/venv/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="1"

WORKSPACE="${WORKSPACE:-/workspace}"
echo "[start] Initializing workspace persistence at $WORKSPACE"

# Ensure workspace directories exist
mkdir -p "$WORKSPACE/models" \
         "$WORKSPACE/output" \
         "$WORKSPACE/datasets" \
         "$WORKSPACE/logs" \
         "$WORKSPACE/training_configs" \
         "$WORKSPACE/training_presets" \
         "$WORKSPACE/embedding_templates" \
         "$WORKSPACE/.cache/huggingface"

# Wire HuggingFace cache
mkdir -p /root/.cache
ln -sfn "$WORKSPACE/.cache/huggingface" /root/.cache/huggingface

# Populate initial templates and configs into workspace if not already present
for dir in training_configs training_presets embedding_templates; do
  if [ -d "/OneTrainer/$dir" ] && [ ! -L "/OneTrainer/$dir" ]; then
    cp -an "/OneTrainer/$dir/." "$WORKSPACE/$dir/" 2>/dev/null || true
    rm -rf "/OneTrainer/$dir"
  fi
  ln -sfn "$WORKSPACE/$dir" "/OneTrainer/$dir"
done

# Link models, datasets, output, and logs
for dir in models output datasets logs; do
  if [ -d "/OneTrainer/$dir" ] && [ ! -L "/OneTrainer/$dir" ]; then
    cp -an "/OneTrainer/$dir/." "$WORKSPACE/$dir/" 2>/dev/null || true
    rm -rf "/OneTrainer/$dir"
  fi
  ln -sfn "$WORKSPACE/$dir" "/OneTrainer/$dir"
done

# Copy WebUI into OneTrainer if needed
if [ -d "/opt/webui" ]; then
  cp -r /opt/webui /OneTrainer/
fi

# --- 1. SSH Daemon (RunPod injects PUBLIC_KEY) ---
if [ -n "${PUBLIC_KEY:-}" ]; then
  mkdir -p /root/.ssh && chmod 700 /root/.ssh
  echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
fi
mkdir -p /run/sshd && /usr/sbin/sshd && echo "[start] SSH daemon running on :22" || echo "[start] WARN: sshd failed to start"

# --- 2. JupyterLab (Port 8888) ---
JUPYTER_TOKEN="${JUPYTER_TOKEN:-}"
nohup jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser \
  --ServerApp.token="$JUPYTER_TOKEN" --ServerApp.password='' \
  --ServerApp.root_dir="$WORKSPACE" \
  --ServerApp.allow_origin='*' --ServerApp.allow_remote_access=True \
  --ServerApp.trust_xheaders=True --ServerApp.disable_check_xsrf=True \
  > /var/log/jupyter.log 2>&1 &
if [ -n "$JUPYTER_TOKEN" ]; then
  echo "[start] JupyterLab running on :8888 (token required)"
else
  echo "[start] JupyterLab running on :8888 (no-auth mode)"
fi

# --- 3. FileBrowser (Port 8081) ---
if command -v filebrowser >/dev/null 2>&1; then
  nohup filebrowser -r "$WORKSPACE" -a 0.0.0.0 -p 8081 --noauth -d /tmp/filebrowser.db \
    > /var/log/filebrowser.log 2>&1 &
  echo "[start] FileBrowser running on :8081"
else
  echo "[start] FileBrowser not installed, skipping :8081"
fi

# --- 4. TensorBoard (Port 6006) ---
nohup tensorboard --logdir="$WORKSPACE/logs" --host=0.0.0.0 --port=6006 \
  > /var/log/tensorboard.log 2>&1 &
echo "[start] TensorBoard running on :6006"

# --- 5. Launch Native OneTrainer WebUI (Port 8080) ---
RUNPOD_HTTP_PORT="${RUNPOD_HTTP_PORT:-8080}"
echo "[start] Launching native OneTrainer WebUI on :$RUNPOD_HTTP_PORT..."
cd /OneTrainer
exec python webui/server.py
