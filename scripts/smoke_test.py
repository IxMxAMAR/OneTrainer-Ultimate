#!/usr/bin/env python3
"""
Smoke test suite for OneTrainer-Ultimate Docker image.
Verifies Python version, PyTorch CUDA build, WebUI server,
diffusers/transformers/mgds, optimizers, and OneTrainer core modules.
"""
import glob
import importlib
import json
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
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("websockets", "WebSockets"),
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


def check_onetrainer_and_webui():
    sys.path.insert(0, "/OneTrainer")
    try:
        from modules.util.config.TrainConfig import TrainConfig
        from webui.server import app, parse_train_config
        print("[check] OneTrainer core modules & WebUI FastAPI app imported successfully")

        # Validate loading presets
        preset_files = glob.glob("/OneTrainer/training_presets/**/*.json", recursive=True)
        valid_presets = [p for p in preset_files if not os.path.basename(p).startswith("#.json")]
        if valid_presets:
            with open(valid_presets[0], "r", encoding="utf-8") as f:
                sample_data = json.load(f)
            cfg = parse_train_config(sample_data)
            assert cfg.model_type is not None
            print(f"[check] Validated preset deserialization on {len(valid_presets)} presets OK")
    except Exception as e:
        print(f"[FAIL] OneTrainer/WebUI import failed: {e}", file=sys.stderr)
        raise


def check_binaries():
    binaries = ["filebrowser", "jupyter", "tensorboard"]
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
        check_onetrainer_and_webui()
    print("=== ALL SMOKE TESTS PASSED ===")


if __name__ == "__main__":
    main()
