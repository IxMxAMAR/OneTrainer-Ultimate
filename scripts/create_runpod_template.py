#!/usr/bin/env python3
"""
Automated RunPod Template Creator for OneTrainer-Ultimate.
Creates or updates the OneTrainer-Ultimate template on your RunPod account.
"""
import argparse
import json
import os
import sys
import urllib.request

RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"

TEMPLATE_PAYLOAD = {
    "name": "OneTrainer-Ultimate",
    "imageName": "ghcr.io/ixmxamar/onetrainer-ultimate:latest",
    "containerDiskInGb": 30,
    "volumeInGb": 100,
    "volumeMountPath": "/workspace",
    "dockerArgs": "",
    "ports": "8080/http,8888/http,8081/http,6006/http,22/tcp",
    "env": [
        {"key": "PORT", "value": "8080"},
        {"key": "HF_HUB_ENABLE_HF_TRANSFER", "value": "1"},
        {"key": "HF_HOME", "value": "/workspace/.cache/huggingface"}
    ],
    "isServerless": False,
    "readme": "# OneTrainer-Ultimate\nNative WebUI for diffusion model training on RunPod."
}


def create_template(api_key: str):
    mutation = """
    mutation SaveTemplate($input: TemplateInput!) {
      saveTemplate(input: $input) {
        id
        name
        imageName
        ports
      }
    }
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    body = {
        "query": mutation,
        "variables": {
            "input": TEMPLATE_PAYLOAD
        }
    }

    req = urllib.request.Request(
        RUNPOD_GRAPHQL_URL,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if "errors" in data:
                print(f"[ERROR] RunPod API error: {json.dumps(data['errors'], indent=2)}")
                return None
            template_info = data.get("data", {}).get("saveTemplate", {})
            print("=" * 60)
            print("✅ Successfully created RunPod Template!")
            print(f"Template Name:  {template_info.get('name')}")
            print(f"Template ID:    {template_info.get('id')}")
            print(f"Container Image: {template_info.get('imageName')}")
            print(f"Exposed Ports:  {template_info.get('ports')}")
            print("=" * 60)
            print("You can now launch pods using this template from: https://www.runpod.io/console/pods")
            return template_info
    except Exception as e:
        print(f"[ERROR] Failed to contact RunPod API: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Create OneTrainer-Ultimate RunPod Template")
    parser.add_argument("--api-key", type=str, default=os.environ.get("RUNPOD_API_KEY"), help="Your RunPod API Key")
    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        api_key = input("Enter your RunPod API Key (from https://www.runpod.io/console/user/settings): ").strip()

    if not api_key:
        print("API Key required. Exiting.")
        sys.exit(1)

    create_template(api_key)


if __name__ == "__main__":
    main()
