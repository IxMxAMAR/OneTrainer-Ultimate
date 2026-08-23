# OneTrainer-Ultimate

[![build](https://github.com/IxMxAMAR/OneTrainer-Ultimate/actions/workflows/build.yml/badge.svg)](https://github.com/IxMxAMAR/OneTrainer-Ultimate/actions/workflows/build.yml)
[![Docker Image](https://img.shields.io/badge/Docker%20Hub-ixmxamar%2Fonetrainer--ultimate-blue?logo=docker)](https://hub.docker.com/r/ixmxamar/onetrainer-ultimate)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0%2Bcu128-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**Batteries-included RunPod Docker image for [OneTrainer](https://github.com/Nerogar/OneTrainer)**: Complete web-accessible Qt6 UI via integrated noVNC, sidecar JupyterLab, FileBrowser, TensorBoard, FlashAttention-2, SageAttention, bitsandbytes, cutting-edge optimizers, and automated `/workspace` volume persistence.

---

## Highlights

- **Web Browser GUI**: Interact with OneTrainer's native Qt6 Desktop GUI directly in your browser over HTTP (port `8080`) via low-latency noVNC with zero client install.
- **Modern Hardware Ready**: Built on **CUDA 12.8 + cuDNN 9 + PyTorch 2.8.0** with full support for **Blackwell (RTX 5090 / sm_120)**, Ada (RTX 4090 / sm_89), Ampere (RTX 3090, A100), and Hopper (H100).
- **Fast Attention Acceleration**: Pre-compiled **FlashAttention-2** (v2.8.3.post1) and **SageAttention** (v2.2.0) baked directly into the image.
- **Advanced Optimizers**: Pre-installed `bitsandbytes` (8-bit / quantization), `Muon`, `Prodigy`, `Prodigy-Plus-Schedule-Free`, `Schedule-Free`, `DAdaptation`, `Lion`, and `adv_optm`.
- **RunPod Persistent Volume Wiring**: Automatically symlinks configs, presets, datasets, models, logs, and outputs to `/workspace` so your training sessions and checkpoints survive pod restarts.
- **Built-in Power Tools**:
  - **JupyterLab** (port `8888`) for interactive Python scripting, dataset slicing, and shell access.
  - **FileBrowser** (port `8081`) for drag-and-drop dataset uploads and checkpoint downloads.
  - **TensorBoard** (port `6006`) live-monitoring loss curves and sample image evaluations in `/workspace/logs`.
  - **OpenSSH** (port `22`) with RunPod `$PUBLIC_KEY` authorization.
  - **Desktop Right-Click Menu**: Launch OneTrainer Qt6, Caption & Masking UI, Model Converter UI, `xterm`, and `nvitop`.

---

## Quick Start on RunPod

### 1. Pod Template Configuration

When creating a custom template or deploying a Pod on RunPod:

- **Container Image**: `ixmxamar/onetrainer-ultimate:latest`
- **Container Disk**: Minimum 20 GB (30+ GB recommended)
- **Volume Disk**: 50–200+ GB (mounted to `/workspace`)

### 2. Port Mappings

Expose the following HTTP / TCP ports in RunPod:

| Port | Protocol | Service | Description |
| :--- | :--- | :--- | :--- |
| **8080** | HTTP | **OneTrainer Web UI** | Web noVNC desktop running OneTrainer GUI |
| **8888** | HTTP | **JupyterLab** | Web terminal, code editor, and file manager |
| **8081** | HTTP | **FileBrowser** | High-speed web file uploader / downloader |
| **6006** | HTTP | **TensorBoard** | Training metrics, loss graphs, and sample logs |
| **22** | TCP | **SSH** | Remote shell & SFTP / VS Code Remote |
| **5900** | TCP | **VNC** | Direct VNC connection (optional) |

### 3. Environment Variables (Optional)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PUBLIC_KEY` | *(None)* | Injected by RunPod for passwordless SSH authentication |
| `JUPYTER_TOKEN` | `""` | Set a token for JupyterLab (leave empty for open access) |
| `VNC_RESOLUTION` | `1920x1080` | Virtual display resolution (e.g. `2560x1440` or `1920x1080`) |
| `RUNPOD_HTTP_PORT`| `8080` | Port for the noVNC web server |

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

## Desktop Environment & Tools

Connecting to port `8080` opens a lightweight, dark-themed X11 environment:

- **OneTrainer Qt6** launches automatically full screen on startup.
- If closed, it automatically restarts, or you can right-click the desktop to access:
  - **OneTrainer (Qt6 Main GUI)**
  - **Caption & Masking Tool** (`caption_ui.py`)
  - **Model Converter UI** (`convert_model_ui.py`)
  - **OneTrainer (CustomTkinter)** (`train_ui_ctk.py`)
  - **Terminal (`xterm`)**
  - **GPU Monitor (`nvitop`)**

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
  -e VNC_RESOLUTION=1920x1080 \
  ixmxamar/onetrainer-ultimate:latest
```

Then open `http://localhost:8080` in your web browser.

---

## CI / CD & Automated Builds

Builds are automatically triggered on push to `main` via GitHub Actions (`.github/workflows/build.yml`):

1. Cleans runner storage and relocates Docker storage to `/mnt`.
2. Builds with Docker Buildx and validates ABI integrity against `constraints.txt`.
3. Runs the internal `smoke_test.py` CI gate (validates PyTorch CUDA, PySide6, diffusers, optimizers, and binaries).
4. Pushes tagged releases (`latest`, commit SHA, `cu128-torch2.8.0`) to Docker Hub.

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
