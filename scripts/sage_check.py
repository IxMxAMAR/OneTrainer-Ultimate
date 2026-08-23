#!/usr/bin/env python3
"""
Validates SageAttention C++ extension loading without requiring a GPU at build time.
"""
import sys

try:
    import sageattention  # noqa: F401
    import sageattention._qattn  # noqa: F401
    print("[sage_check] SageAttention C++ extension imported successfully.")
    sys.exit(0)
except ImportError as e:
    err_str = str(e).lower()
    if "no cuda-capable device is detected" in err_str or "cuda error" in err_str or "driver" in err_str:
        print(f"[sage_check] SageAttention C++ extension found (GPU not present at build): {e}")
        sys.exit(0)
    print(f"[sage_check] SageAttention import failed (ABI mismatch or missing symbols): {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"[sage_check] Unexpected error importing SageAttention: {e}", file=sys.stderr)
    sys.exit(1)
