import os
import json
import asyncio
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = os.getenv("MODEL_PATH",   "/app/model/resnet18_cats.pth")
DATA_DIR     = os.getenv("DATA_DIR",     "/app/data")
CAT_DIR      = os.getenv("CAT_DIR",      "/app/data/cat")
NOT_CAT_DIR  = os.getenv("NOT_CAT_DIR",  "/app/data/not_cat")
DEFAULT_EPOCHS     = int(os.getenv("DEFAULT_EPOCHS",     5))
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", 32))
DEFAULT_LR         = float(os.getenv("DEFAULT_LR",       0.001))

app = FastAPI(title="Cat Recognition API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model state ─────────────────────────────────────────────────────────
model       = None   # loaded PyTorch model
model_ready = False  # True once weights are loaded

# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    global model, model_ready
    Path(CAT_DIR).mkdir(parents=True, exist_ok=True)
    Path(NOT_CAT_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

    if Path(MODEL_PATH).exists():
        print(f"Model found at {MODEL_PATH}, loading...")
        _load_model()
    else:
        print("No model found. Train first via the dashboard.")

def _load_model():
    """Load model weights from disk. Called on startup and after training."""
    global model, model_ready
    # TODO: implement in Phase 2
    model_ready = False
    print("Model loading not yet implemented.")

# ── Helpers ────────────────────────────────────────────────────────────────────
def count_images(directory: str) -> int:
    p = Path(directory)
    if not p.exists():
        return 0
    return len([f for f in p.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}])

# ── Schemas ────────────────────────────────────────────────────────────────────
class FetchRequest(BaseModel):
    cats:     int = 200
    not_cats: int = 200

class TrainRequest(BaseModel):
    epochs:     int   = DEFAULT_EPOCHS
    batch_size: int   = DEFAULT_BATCH_SIZE
    lr:         float = DEFAULT_LR

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":      "ok",
        "model_ready": model_ready,
        "model_path":  MODEL_PATH,
        "dataset": {
            "cat":     count_images(CAT_DIR),
            "not_cat": count_images(NOT_CAT_DIR),
        }
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accept an image, return cat/not_cat probabilities."""
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first.")
    # TODO: implement in Phase 2
    raise HTTPException(status_code=501, detail="Not implemented yet.")


@app.post("/data/fetch")
def fetch_data(req: FetchRequest):
    """Download cat and not-cat images using icrawler. SSE stream of progress."""
    def generate():
        # TODO: implement in Phase 1
        yield f"data: {json.dumps({'type': 'error', 'text': 'Not implemented yet.'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/train")
def train(req: TrainRequest):
    """Fine-tune ResNet18. SSE stream of per-epoch loss and accuracy."""
    def generate():
        # TODO: implement in Phase 1
        yield f"data: {json.dumps({'type': 'error', 'text': 'Not implemented yet.'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/train/runs")
def list_runs():
    """Return all past training runs with metrics."""
    # TODO: implement in Phase 1 (store in JSON file or simple DB)
    return []