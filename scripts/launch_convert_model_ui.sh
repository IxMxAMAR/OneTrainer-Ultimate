#!/usr/bin/env bash
export DISPLAY="${DISPLAY:-:1}"
export QT_QPA_PLATFORM="xcb"
export QT_X11_NO_MITSHM="1"
export XDG_RUNTIME_DIR="/tmp/runtime-root"
export PATH="/opt/venv/bin:$PATH"

cd /OneTrainer
exec python scripts/convert_model_ui.py "$@"
