#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Ensure OneTrainer is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from webui.server import run

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "127.0.0.1")
    run(host=host, port=port)
