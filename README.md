# OneTrainer-Ultimate

[![build](https://github.com/IxMxAMAR/OneTrainer-Ultimate/actions/workflows/build.yml/badge.svg)](https://github.com/IxMxAMAR/OneTrainer-Ultimate/actions/workflows/build.yml)
[![Docker Image](https://img.shields.io/badge/Docker%20Hub-ixmxamar%2Fonetrainer--ultimate-blue?logo=docker)](https://hub.docker.com/r/ixmxamar/onetrainer-ultimate)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0%2Bcu128-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**Batteries-included RunPod Docker image for [OneTrainer](https://github.com/Nerogar/OneTrainer)**: Native, web-first training UI (FastAPI + WebSockets + responsive dark SPA) designed for high-performance cloud training on RunPod. Includes sidecar JupyterLab, FileBrowser, TensorBoard, FlashAttention-2, SageAttention, bitsandbytes, cutting-edge optimizers, and automated `/workspace` volume persistence.

---

## Highlights

- **Native WebUI**: Full-featured, responsive browser interface running on port `8080` (no VNC client or desktop streaming required).
- **Modern Hardware Ready**: Built on **CUDA 12.8 + cuDNN 9 + PyTorch 2.8.0** with full support for **Blackwell (RTX 5090 / sm_120)**, Ada (RTX 4090 / sm_89), Ampere (RTX 3090, A100), and Hopper (H100).
- **Fast Attention Acceleration**: Pre-compiled **FlashAttention-2** (v2.8.3.post1) and **SageAttention** (v2.2.0) baked directly into the image.
- **Advanced Optimizers**: Pre-installed `bitsandbytes` (8-bit / quantization), `Muon`, `Prodigy`, `Prodigy-Plus-Schedule-Free`, `Schedule-Free`, `DAdaptation`, `Lion`, and `adv_optm`.
- **RunPod Persistent Volume Wiring**: Automatically symlinks configs, presets, datasets, models, logs, and outputs to `/workspace` so your training sessions and checkpoints survive pod restarts.
- **Built-in Power Tools**:
  - **JupyterLab** (port `8888`) for interactive Python scripting, dataset slicing, and shell access.
  - **FileBrowser** (port `8081`) for drag-and-drop dataset uploads and checkpoint downloads.
  - **TensorBoard** (port `6006`) live-monitoring loss curves and sample image evaluations in `/workspace/logs`.
  - **OpenSSH** (port `22`) with RunPod `$PUBLIC_KEY` authorization.

---

## Quick Start on RunPod

### 1. Pod Template Configuration

When creating a custom template or deploying a Pod on RunPod:

- **Container Image**: `ixmxamar/onetrainer-ultimate:latest` (or `ghcr.io/ixmxamar/onetrainer-ultimate:latest`)
- **Container Disk**: Minimum 20 GB (30+ GB recommended)
- **Volume Disk**: 50–200+ GB (mounted to `/workspace`)

### 2. Port Mappings

Expose the following HTTP / TCP ports in RunPod:

| Port | Protocol | Service | Description |
| :--- | :--- | :--- | :--- |
| **8080** | HTTP | **OneTrainer WebUI** | Native browser UI for diffusion training |
| **8888** | HTTP | **JupyterLab** | Web terminal, code editor, and file manager |
| **8081** | HTTP | **FileBrowser** | High-speed web file uploader / downloader |
| **6006** | HTTP | **TensorBoard** | Training metrics, loss graphs, and sample logs |
| **22** | TCP | **SSH** | Remote shell & SFTP / VS Code Remote |

### 3. Environment Variables (Optional)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PUBLIC_KEY` | *(None)* | Injected by RunPod for passwordless SSH authentication |
| `JUPYTER_TOKEN` | `""` | Set a token for JupyterLab (leave empty for open access) |
| `PORT` | `8080` | Port for the OneTrainer WebUI |

---

## Directory Structure & Persistence

When a network volume is attached to `/workspace`, the container automatically links and preserves:

```text
/workspace/
├── datasets/             # Place training datasets and image folders here
├── models/               # Base models (SD 1.5, SDXL, Flux, Cascade, etc.)
├── output/               # Saved LoRA weights, checkpoints, and export files
├── logs/                 # TensorBoard event logs (served on :6006)
├── training_configs/     # Saved training configurations (.json)
├── training_presets/     # Custom training presets
└── embedding_templates/  # Textual inversion embedding templates
```

---

## Running Locally via Docker

```bash
docker run --gpus all -it --rm \
  -p 8080:8080 \
  -p 8888:8888 \
  -p 8081:8081 \
  -p 6006:6006 \
  -p 2222:22 \
  -v $(pwd)/workspace:/workspace \
  ixmxamar/onetrainer-ultimate:latest
```

Then open `http://localhost:8080` in your web browser.

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
