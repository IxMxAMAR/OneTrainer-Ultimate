#!/usr/bin/env bash
# =============================================================================
# RunPod Pod Entrypoint: OneTrainer Ultimate
# Sets up workspace persistence, starts background daemons (SSH, JupyterLab,
# FileBrowser, TensorBoard), initializes virtual display (Xvfb + Openbox +
# x11vnc + noVNC), and runs OneTrainer Qt6 GUI.
# =============================================================================
set -u
export PATH="/opt/venv/bin:$PATH"
export DISPLAY="${DISPLAY:-:1}"
export VNC_RESOLUTION="${VNC_RESOLUTION:-1920x1080}"
export QT_QPA_PLATFORM="xcb"
export QT_X11_NO_MITSHM="1"
export XDG_RUNTIME_DIR="/tmp/runtime-root"

# Hugging Face cache persistence and high-speed transfer
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="1"

mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

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
echo "[start] TensorBoard running on :6006 (watching $WORKSPACE/logs)"

# --- 5. Virtual Display Stack (Xvfb + Openbox + x11vnc + noVNC on Port 8080) ---
echo "[start] Starting Xvfb virtual display on $DISPLAY ($VNC_RESOLUTION x 24)..."
Xvfb "$DISPLAY" -screen 0 "${VNC_RESOLUTION}x24" -ac +extension GLX +render -noreset \
  > /var/log/xvfb.log 2>&1 &
sleep 1

# Setup Openbox window manager
mkdir -p /root/.config/openbox
if [ -f /opt/scripts/openbox-rc.xml ]; then
  cp /opt/scripts/openbox-rc.xml /root/.config/openbox/rc.xml
fi
if [ -f /opt/scripts/openbox-menu.xml ]; then
  cp /opt/scripts/openbox-menu.xml /root/.config/openbox/menu.xml
fi

openbox-session > /var/log/openbox.log 2>&1 &
sleep 1

# Dark background
xsetroot -solid "#1a1b26" 2>/dev/null || true

# Start x11vnc server on port 5900
x11vnc -display "$DISPLAY" -nopw -listen 127.0.0.1 -xkb -ncache 10 -ncache_cr -forever -shared -repeat -rfbport 5900 \
  > /var/log/x11vnc.log 2>&1 &
sleep 1

# Start websockify / noVNC on port 8080 (Primary RunPod HTTP Web UI)
RUNPOD_HTTP_PORT="${RUNPOD_HTTP_PORT:-8080}"
echo "[start] Launching noVNC Web UI on :$RUNPOD_HTTP_PORT..."
websockify --web /usr/share/novnc "$RUNPOD_HTTP_PORT" 127.0.0.1:5900 \
  > /var/log/novnc.log 2>&1 &
sleep 1

echo "[start] All daemons online. Launching OneTrainer Qt6 GUI..."

# --- 6. OneTrainer UI Supervisor Loop ---
cd /OneTrainer

while true; do
  echo "[start] Starting OneTrainer UI process..."
  python scripts/train_ui_qt.py
  EXIT_CODE=$?
  echo "[start] OneTrainer exited with code $EXIT_CODE. Restarting in 2 seconds (or use desktop right-click menu)..."
  sleep 2
done
