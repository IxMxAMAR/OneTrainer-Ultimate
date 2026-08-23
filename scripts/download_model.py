#!/usr/bin/env python3
"""
CLI & UI Model Downloader for OneTrainer-Ultimate.
Downloads base models and checkpoints from Hugging Face directly into /workspace/models
with accelerated transfers (hf_transfer / aria2).
"""
import argparse
import os
import sys

from huggingface_hub import snapshot_download, hf_hub_download

POPULAR_MODELS = {
    "flux-dev": {
        "repo_id": "black-forest-labs/FLUX.1-dev",
        "description": "FLUX.1 [dev] 12B Flow-Match Diffusion model (requires HF Token)",
        "subfolder": None,
    },
    "flux-schnell": {
        "repo_id": "black-forest-labs/FLUX.1-schnell",
        "description": "FLUX.1 [schnell] 12B Fast 4-step model",
        "subfolder": None,
    },
    "sdxl-base": {
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "description": "Stable Diffusion XL 1.0 Base",
        "subfolder": None,
    },
    "sd15": {
        "repo_id": "runwayml/stable-diffusion-v1-5",
        "description": "Stable Diffusion 1.5 Base",
        "subfolder": None,
    },
    "sd35-large": {
        "repo_id": "stabilityai/stable-diffusion-3.5-large",
        "description": "Stable Diffusion 3.5 Large (requires HF Token)",
        "subfolder": None,
    },
    "sana-1600m": {
        "repo_id": "Efficient-Large-Model/Sana_1600M_1024px",
        "description": "Sana 1.6B Linear Attention Diffusion Model",
        "subfolder": None,
    },
    "hunyuan-video": {
        "repo_id": "tencent/HunyuanVideo",
        "description": "HunyuanVideo Foundation Model",
        "subfolder": None,
    },
}


def download_model(repo_id: str, local_dir: str, token: str | None = None, allow_patterns: list[str] | None = None):
    print(f"[download] Starting download of {repo_id} to {local_dir}...")
    os.makedirs(local_dir, exist_ok=True)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=token or os.environ.get("HF_TOKEN"),
        allow_patterns=allow_patterns,
        resume_download=True,
    )
    print(f"[download] Successfully downloaded {repo_id} to {local_dir}")


def main():
    parser = argparse.ArgumentParser(description="OneTrainer Model Downloader")
    parser.add_argument("--model", choices=list(POPULAR_MODELS.keys()), help="Popular preset model to download")
    parser.add_argument("--repo-id", type=str, help="Any HuggingFace repo ID (e.g. organization/model-name)")
    parser.add_argument("--dest", type=str, default="/workspace/models", help="Destination directory (default: /workspace/models)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token for gated models")
    parser.add_argument("--list", action="store_true", help="List available model presets")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable model presets:")
        for k, v in POPULAR_MODELS.items():
            print(f"  {k:<16} - {v['description']} ({v['repo_id']})")
        print("\nYou can also download any model with --repo-id <huggingface_repo_id>\n")
        return

    if args.model:
        preset = POPULAR_MODELS[args.model]
        target_dir = os.path.join(args.dest, args.model)
        download_model(preset["repo_id"], target_dir, args.token)
    elif args.repo_id:
        model_folder_name = args.repo_id.replace("/", "--")
        target_dir = os.path.join(args.dest, model_folder_name)
        download_model(args.repo_id, target_dir, args.token)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
