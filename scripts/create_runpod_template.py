#!/usr/bin/env python3
"""
Automated RunPod Template Creator for OneTrainer-Ultimate.
Creates or updates the OneTrainer-Ultimate template on your RunPod account.
"""
import argparse
import os
import sys
import runpod


def create_template(api_key: str):
    runpod.api_key = api_key
    try:
        template = runpod.create_template(
            name="OneTrainer-Ultimate",
            image_name="ghcr.io/ixmxamar/onetrainer-ultimate:latest",
            container_disk_in_gb=30,
            volume_in_gb=100,
            volume_mount_path="/workspace",
            ports="8080/http,8888/http,8081/http,6006/http,22/tcp",
            env={
                "PORT": "8080",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "HF_HOME": "/workspace/.cache/huggingface"
            },
            is_serverless=False
        )
        print("=" * 60)
        print("✅ Successfully created RunPod Template!")
        print(f"Template Name:   {template.get('name')}")
        print(f"Template ID:     {template.get('id')}")
        print(f"Container Image: {template.get('imageName')}")
        print(f"Exposed Ports:   {template.get('ports')}")
        print("=" * 60)
        print("Launch Pods directly at: https://www.runpod.io/console/gpu-cloud")
        return template
    except Exception as e:
        print(f"[ERROR] Failed to create template: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Create OneTrainer-Ultimate RunPod Template")
    parser.add_argument("--api-key", type=str, default=os.environ.get("RUNPOD_API_KEY"), help="Your RunPod API Key")
    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        api_key = input("Enter your RunPod API Key: ").strip()

    if not api_key:
        print("API Key required. Exiting.")
        sys.exit(1)

    create_template(api_key)


if __name__ == "__main__":
    main()
