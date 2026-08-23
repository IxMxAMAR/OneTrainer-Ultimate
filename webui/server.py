import asyncio
import base64
import glob
import io
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure OneTrainer root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from modules.util import create
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.DataType import DataType
from modules.util.TrainProgress import TrainProgress
from modules.util.torch_util import torch_gc
import torch

app = FastAPI(title="OneTrainer WebUI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class TrainingState:
    def __init__(self):
        self.config: TrainConfig = TrainConfig.default_values()
        self.is_training: bool = False
        self.status: str = "Idle"
        self.epoch: int = 0
        self.max_epoch: int = 1
        self.step: int = 0
        self.max_step: int = 100
        self.total_step: int = 0
        self.eta_str: str = ""
        self.loss_history: List[Dict[str, Any]] = []
        self.sample_images: List[Dict[str, Any]] = []
        self.logs: List[str] = []
        self.trainer = None
        self.training_thread: Optional[threading.Thread] = None
        self.training_commands: Optional[TrainCommands] = None
        self.training_callbacks: Optional[TrainCallbacks] = None
        self.start_time: Optional[float] = None
        self.start_total_steps: Optional[int] = None
        self.stop_requested: bool = False

    def reset_progress(self):
        self.epoch = 0
        self.step = 0
        self.total_step = 0
        self.eta_str = "Estimating..."
        self.loss_history = []
        self.sample_images = []
        self.logs = []
        self.stop_requested = False

    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.logs.append(entry)
        if len(self.logs) > 500:
            self.logs.pop(0)

    def to_status_dict(self) -> Dict[str, Any]:
        gpu_info = {}
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                gpu_name = torch.cuda.get_device_name(0)
                gpu_info = {
                    "name": gpu_name,
                    "allocated_gb": round(allocated, 2),
                    "reserved_gb": round(reserved, 2),
                    "total_gb": round(total, 2),
                }
            except Exception:
                pass

        return {
            "is_training": self.is_training,
            "status": self.status,
            "epoch": self.epoch,
            "max_epoch": self.max_epoch,
            "step": self.step,
            "max_step": self.max_step,
            "total_step": self.total_step,
            "eta": self.eta_str,
            "gpu": gpu_info,
            "recent_loss": self.loss_history[-1]["loss"] if self.loss_history else None,
            "sample_count": len(self.sample_images),
        }


state = TrainingState()
active_websockets: List[WebSocket] = []


async def broadcast_status():
    if not active_websockets:
        return
    data = json.dumps({"type": "status", "data": state.to_status_dict()})
    for ws in list(active_websockets):
        try:
            await ws.send_text(data)
        except Exception:
            if ws in active_websockets:
                active_websockets.remove(ws)


async def broadcast_event(event_type: str, payload: Any):
    if not active_websockets:
        return
    data = json.dumps({"type": event_type, "data": payload})
    for ws in list(active_websockets):
        try:
            await ws.send_text(data)
        except Exception:
            if ws in active_websockets:
                active_websockets.remove(ws)


@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
async def get_status():
    return state.to_status_dict()


@app.get("/api/config")
async def get_current_config():
    return state.config.to_pack_dict(secrets=True)


@app.post("/api/config")
async def update_current_config(config_data: Dict[str, Any]):
    try:
        current_dict = state.config.to_pack_dict(secrets=True)
        # Deep merge updates
        def update_deep(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    update_deep(d[k], v)
                else:
                    d[k] = v
        update_deep(current_dict, config_data)
        state.config = TrainConfig.from_dict(current_dict)
        return {"status": "ok", "config": state.config.to_pack_dict(secrets=True)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/presets")
async def list_presets():
    presets_dir = SCRIPT_DIR / "training_presets"
    results = {}
    if presets_dir.exists():
        for category in sorted(os.listdir(presets_dir)):
            cat_path = presets_dir / category
            if cat_path.is_dir():
                results[category] = []
                for file in sorted(os.listdir(cat_path)):
                    if file.endswith(".json"):
                        name = file[:-5].lstrip("#")
                        results[category].append({"name": name, "filename": file, "category": category})
    return results


@app.get("/api/preset/{category}/{filename}")
async def load_preset(category: str, filename: str):
    preset_file = SCRIPT_DIR / "training_presets" / category / filename
    if not preset_file.exists():
        raise HTTPException(status_code=404, detail="Preset not found")
    try:
        with open(preset_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        state.config = TrainConfig.from_dict(data)
        state.log(f"Loaded preset: {category}/{filename}")
        return {"status": "ok", "config": state.config.to_pack_dict(secrets=True)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/saved_configs")
async def list_saved_configs():
    workspace = Path(state.config.workspace_dir or "/workspace")
    configs_dir = workspace / "training_configs"
    if not configs_dir.exists():
        configs_dir = SCRIPT_DIR / "training_configs"
    configs = []
    if configs_dir.exists():
        for f in sorted(glob.glob(str(configs_dir / "*.json"))):
            configs.append(Path(f).name)
    return configs


@app.post("/api/save_config")
async def save_config_file(payload: Dict[str, Any]):
    name = payload.get("filename", "config.json")
    if not name.endswith(".json"):
        name += ".json"
    workspace = Path(state.config.workspace_dir or "/workspace")
    target_dir = workspace / "training_configs"
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / name
    try:
        config_dict = state.config.to_pack_dict(secrets=False)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)
        state.log(f"Saved configuration to {file_path}")
        return {"status": "ok", "path": str(file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/load_config")
async def load_config_file(payload: Dict[str, Any]):
    name = payload.get("filename", "")
    workspace = Path(state.config.workspace_dir or "/workspace")
    file_path = workspace / "training_configs" / name
    if not file_path.exists():
        file_path = SCRIPT_DIR / "training_configs" / name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Config {name} not found")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        state.config = TrainConfig.from_dict(data)
        state.log(f"Loaded configuration from {file_path}")
        return {"status": "ok", "config": state.config.to_pack_dict(secrets=True)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Training Execution Engine ---

def _on_update_train_progress(train_progress: TrainProgress, max_step: int, max_epoch: int):
    state.epoch = train_progress.epoch
    state.max_epoch = max_epoch
    state.step = train_progress.epoch_step
    state.max_step = max_step
    state.total_step = train_progress.epoch * max_step + train_progress.epoch_step

    if state.start_time is not None and state.start_total_steps is not None:
        spent = time.monotonic() - state.start_time
        steps_done = state.total_step - state.start_total_steps
        remaining = (max_epoch - state.epoch - 1) * max_step + (max_step - state.step)
        if steps_done > 5 and remaining >= 0:
            rate = steps_done / max(spent, 0.001)
            eta_sec = remaining / max(rate, 0.001)
            mins, secs = divmod(int(eta_sec), 60)
            hours, mins = divmod(mins, 60)
            state.eta_str = f"{hours:02d}:{mins:02d}:{secs:02d}" if hours > 0 else f"{mins:02d}:{secs:02d}"
        else:
            state.eta_str = "Estimating..."

    # Record loss if available in progress
    if hasattr(train_progress, "loss") and train_progress.loss is not None:
        loss_val = float(train_progress.loss)
        state.loss_history.append({"step": state.total_step, "loss": round(loss_val, 5)})
        if len(state.loss_history) > 2000:
            state.loss_history.pop(0)

    asyncio.run(broadcast_status())


def _on_update_status(status_msg: str):
    state.status = status_msg
    state.log(status_msg)
    asyncio.run(broadcast_status())


def _on_sample_output(sampler_output):
    if hasattr(sampler_output, "images") and sampler_output.images:
        for img in sampler_output.images:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            sample_entry = {
                "timestamp": time.strftime("%H:%M:%S"),
                "step": state.total_step,
                "epoch": state.epoch,
                "image": f"data:image/jpeg;base64,{b64_str}",
            }
            state.sample_images.append(sample_entry)
            asyncio.run(broadcast_event("sample", sample_entry))


def _training_worker():
    state.is_training = True
    state.status = "Initializing..."
    state.reset_progress()
    state.log("Training run starting...")

    state.training_callbacks = TrainCallbacks(
        on_update_train_progress=_on_update_train_progress,
        on_update_status=_on_update_status,
        on_sample_default=_on_sample_output,
        on_sample_custom=_on_sample_output,
    )
    state.training_commands = TrainCommands()

    try:
        state.trainer = create.create_trainer(
            state.config,
            state.training_callbacks,
            state.training_commands,
            reattach=False,
        )
        state.trainer.start()
        state.start_total_steps = state.epoch * state.max_step + state.step
        state.start_time = time.monotonic()
        state.status = "Training"
        state.trainer.train()
    except Exception as e:
        traceback.print_exc()
        state.status = f"Error: {e}"
        state.log(f"Training error: {e}")
    finally:
        if state.trainer:
            try:
                state.trainer.end()
            except Exception:
                pass
            del state.trainer
            state.trainer = None

        state.is_training = False
        state.training_commands = None
        if not state.status.startswith("Error"):
            state.status = "Stopped"
        state.log("Training run finished.")
        torch.clear_autocast_cache()
        torch_gc()
        asyncio.run(broadcast_status())


@app.post("/api/train/start")
async def start_training():
    if state.is_training:
        raise HTTPException(status_code=400, detail="Training already in progress")
    state.training_thread = threading.Thread(target=_training_worker, daemon=True)
    state.training_thread.start()
    return {"status": "started"}


@app.post("/api/train/stop")
async def stop_training():
    if not state.is_training:
        return {"status": "not_training"}
    state.status = "Stopping..."
    state.stop_requested = True
    if state.training_commands:
        state.training_commands.stop()
    return {"status": "stopping"}


@app.post("/api/train/sample_now")
async def sample_now():
    if state.training_commands:
        state.training_commands.sample_default()
        state.log("Triggered manual sample generation")
        return {"status": "sample_requested"}
    raise HTTPException(status_code=400, detail="Training is not running")


@app.post("/api/train/save_now")
async def save_now():
    if state.training_commands:
        state.training_commands.save()
        state.log("Triggered manual checkpoint save")
        return {"status": "save_requested"}
    raise HTTPException(status_code=400, detail="Training is not running")


@app.get("/api/loss_history")
async def get_loss_history():
    return state.loss_history


@app.get("/api/samples")
async def get_samples():
    return state.sample_images


@app.get("/api/logs")
async def get_logs():
    return state.logs


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    # Send initial snapshot
    await websocket.send_text(json.dumps({
        "type": "init",
        "status": state.to_status_dict(),
        "loss_history": state.loss_history,
        "logs": state.logs,
    }))
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)


def run(host: str = "0.0.0.0", port: int = 8080):
    import uvicorn
    print("=" * 60)
    print(f"OneTrainer WebUI starting on http://{host}:{port}")
    print("=" * 60)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    run(host="0.0.0.0", port=port)
