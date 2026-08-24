import contextlib
import os
from collections.abc import Callable

import huggingface_hub
from huggingface_hub import constants as hf_constants


def configure_hub(
        token: str = "",
        offline_mode: bool = False,
        cache_dir: str = "",
        on_status: Callable[[str], None] | None = None,
):
    os.environ.setdefault("HF_XET_FIXED_DOWNLOAD_CONCURRENCY", "8")

    hf_constants.HF_HUB_OFFLINE = offline_mode
    if cache_dir:
        hf_constants.HF_HUB_CACHE = cache_dir

    if not offline_mode and token != "":
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        if on_status is not None:
            on_status("logging into Hugging Face")
        with contextlib.suppress(Exception):
            huggingface_hub.login(token=token)
