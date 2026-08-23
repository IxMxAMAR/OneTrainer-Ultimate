#!/usr/bin/env python3
"""
Smoke test suite for OneTrainer-Ultimate Docker image.
Verifies Python version, PyTorch CUDA build, PySide6 Qt bindings,
diffusers/transformers/mgds, optimizers, and OneTrainer core modules.
"""
import importlib
import os
import shutil
import sys


def check_python():
    print(f"[check] Python: {sys.version}")
    assert sys.version_info >= (3, 10), f"Python version too old: {sys.version}"
    assert sys.version_info < (3, 14), f"Python version too new: {sys.version}"


def check_torch():
    import torch
    print(f"[check] PyTorch: {torch.__version__} (CUDA: {torch.version.cuda})")
    assert "+cu" in torch.__version__ or torch.cuda.is_available() or torch.version.cuda is not None, \
        f"Torch CUDA build missing: {torch.__version__}"


def check_core_packages():
    required_modules = [
        ("PySide6", "PySide6"),
        ("customtkinter", "customtkinter"),
        ("transformers", "transformers"),
        ("diffusers", "diffusers"),
        ("accelerate", "accelerate"),
        ("safetensors", "safetensors"),
        ("mgds", "mgds"),
        ("bitsandbytes", "bitsandbytes"),
        ("tensorboard", "tensorboard"),
        ("scipy", "scipy"),
        ("PIL", "Pillow"),
        ("cv2", "opencv"),
        ("numpy", "numpy"),
        ("prodigyopt", "prodigyopt"),
        ("dadaptation", "dadaptation"),
        ("schedulefree", "schedulefree"),
        ("lion_pytorch", "lion_pytorch"),
        ("open_clip", "open_clip"),
        ("gguf", "gguf"),
    ]

    for mod_name, label in required_modules:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"[check] {label} ({mod_name}) OK: {ver}")
        except Exception as e:
            print(f"[FAIL] Missing required module {label} ({mod_name}): {e}", file=sys.stderr)
            raise


def check_onetrainer_imports():
    sys.path.insert(0, "/OneTrainer")
    try:
        from modules.util.config.TrainConfig import TrainConfig
        from modules.ui.PySide6TrainUIView import PySide6TrainView
        print("[check] OneTrainer core modules and Qt6 UI view import OK")
    except Exception as e:
        print(f"[FAIL] OneTrainer import failed: {e}", file=sys.stderr)
        raise


def check_binaries():
    binaries = ["filebrowser", "Xvfb", "x11vnc", "openbox", "websockify", "jupyter", "tensorboard"]
    for b in binaries:
        path = shutil.which(b)
        assert path is not None, f"Required binary missing in PATH: {b}"
        print(f"[check] Binary {b}: {path}")


def main():
    print("=== OneTrainer-Ultimate Smoke Test Suite ===")
    check_python()
    check_torch()
    check_core_packages()
    check_binaries()
    if os.path.exists("/OneTrainer"):
        check_onetrainer_imports()
    print("=== ALL SMOKE TESTS PASSED ===")


if __name__ == "__main__":
    main()
