import os
import json
import time
import re
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.getenv("MODEL_PATH",  "/app/model/resnet18_cats.pth")
DATA_DIR    = os.getenv("DATA_DIR",    "/app/data")
CAT_DIR     = os.getenv("CAT_DIR",     "/app/data/cat")
NOT_CAT_DIR = os.getenv("NOT_CAT_DIR", "/app/data/not_cat")
RUNS_FILE   = os.getenv("RUNS_FILE",   "/app/data/runs.json")

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
_model           = None
_model_ready     = False
_model_lock      = threading.Lock()
_cancel_training = False

# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    Path(CAT_DIR).mkdir(parents=True, exist_ok=True)
    Path(NOT_CAT_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    if not Path(RUNS_FILE).exists():
        Path(RUNS_FILE).write_text("[]")

    if Path(MODEL_PATH).exists():
        print(f"Model found at {MODEL_PATH}, loading...")
        _load_model()
    else:
        print("No model found. Use the dashboard to fetch data and train.")

# ── Helpers ────────────────────────────────────────────────────────────────────
def count_images(directory: str) -> int:
    p = Path(directory)
    if not p.exists():
        return 0
    return len([f for f in p.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}])


def load_runs() -> list:
    try:
        return json.loads(Path(RUNS_FILE).read_text())
    except:
        return []


def save_run(run: dict):
    runs = load_runs()
    runs.append(run)
    Path(RUNS_FILE).write_text(json.dumps(runs, indent=2))


def _load_model():
    global _model, _model_ready
    try:
        import torch
        import torchvision.models as models

        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()

        with _model_lock:
            _model = model
            _model_ready = True
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")


def _preprocess_image(image_bytes: bytes):
    """Resize, normalize, return tensor ready for inference."""
    from PIL import Image
    import torchvision.transforms as transforms
    import torch
    import io

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(img).unsqueeze(0)  # add batch dim


# ── Schemas ────────────────────────────────────────────────────────────────────
class TrainRequest(BaseModel):
    epochs:     int   = DEFAULT_EPOCHS
    batch_size: int   = DEFAULT_BATCH_SIZE
    lr:         float = DEFAULT_LR

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_ready":  _model_ready,
        "model_path":   MODEL_PATH,
        "dataset": {
            "cat":     count_images(CAT_DIR),
            "not_cat": count_images(NOT_CAT_DIR),
        }
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not _model_ready:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first.")

    import torch
    import torch.nn.functional as F

    image_bytes = await file.read()
    tensor = _preprocess_image(image_bytes)

    with _model_lock:
        with torch.no_grad():
            logits = _model(tensor)
            probs  = F.softmax(logits, dim=1)[0]

    cat_prob     = float(probs[0])
    not_cat_prob = float(probs[1])
    prediction   = "cat" if cat_prob > not_cat_prob else "not_cat"
    confidence   = max(cat_prob, not_cat_prob)

    return {
        "prediction":   prediction,
        "confidence":   round(confidence, 4),
        "probabilities": {
            "cat":     round(cat_prob, 4),
            "not_cat": round(not_cat_prob, 4),
        }
    }





@app.post("/train")
def train(req: TrainRequest):
    """Fine-tune ResNet18. SSE stream of per-epoch metrics."""

    def generate():
        import torch
        import torch.nn as nn
        import torchvision.models as models
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader
        from torchvision.datasets import ImageFolder

        global _cancel_training
        _cancel_training = False

        yield f"data: {json.dumps({'type': 'status', 'text': 'Checking dataset...'})}\n\n"

        cat_count     = count_images(CAT_DIR)
        not_cat_count = count_images(NOT_CAT_DIR)

        if cat_count < 10 or not_cat_count < 10:
            yield f"data: {json.dumps({'type': 'error', 'text': f'Not enough images. Need at least 10 per class. Got: cat={cat_count}, not_cat={not_cat_count}'})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'status', 'text': f'Dataset: {cat_count} cats, {not_cat_count} not-cats. Loading...'})}\n\n"

        # ── Transforms ──────────────────────────────────────────────────────────
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # ── Dataset ─────────────────────────────────────────────────────────────
        # ImageFolder expects: DATA_DIR/cat/... and DATA_DIR/not_cat/...
        try:
            full_dataset = ImageFolder(root=DATA_DIR, transform=train_transform)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': f'Failed to load dataset: {str(e)}'})}\n\n"
            return

        # 80/20 train/val split
        total     = len(full_dataset)
        val_size  = max(1, int(total * 0.2))
        train_size = total - val_size
        train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        val_ds.dataset = ImageFolder(root=DATA_DIR, transform=val_transform)

        train_loader = DataLoader(train_ds, batch_size=req.batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=req.batch_size, shuffle=False, num_workers=0)

        yield f"data: {json.dumps({'type': 'status', 'text': f'Train: {train_size} images, Val: {val_size} images. Building model...'})}\n\n"

        # ── Model ────────────────────────────────────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        yield f"data: {json.dumps({'type': 'status', 'text': f'Using device: {str(device)}'})}\n\n"

        model = models.resnet18(weights="IMAGENET1K_V1")
        # Freeze all layers except the final FC
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: cat, not_cat
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=req.lr)

        # ── Training loop ────────────────────────────────────────────────────────
        start_time  = time.time()
        best_val_acc = 0.0
        history      = []

        yield f"data: {json.dumps({'type': 'status', 'text': f'Starting training for {req.epochs} epochs...'})}\n\n"

        for epoch in range(req.epochs):
            # Train
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                if _cancel_training:
                    yield f"data: {json.dumps({'type': 'cancelled', 'text': 'Training cancelled.'})}\n\n"
                    return

                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss    += loss.item()
                _, predicted   = outputs.max(1)
                train_total   += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

                # Progress within epoch
                if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
                    yield f"data: {json.dumps({'type': 'batch', 'epoch': epoch+1, 'epochs': req.epochs, 'batch': batch_idx+1, 'batches': len(train_loader)})}\n\n"

            train_acc  = train_correct / train_total
            train_loss = train_loss / len(train_loader)

            # Validate
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs        = model(images)
                    loss           = criterion(outputs, labels)
                    val_loss      += loss.item()
                    _, predicted   = outputs.max(1)
                    val_total     += labels.size(0)
                    val_correct   += predicted.eq(labels).sum().item()

            val_acc  = val_correct / val_total
            val_loss = val_loss / len(val_loader)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), MODEL_PATH)

            epoch_data = {
                "epoch":      epoch + 1,
                "epochs":     req.epochs,
                "train_loss": round(train_loss, 4),
                "train_acc":  round(train_acc,  4),
                "val_loss":   round(val_loss,   4),
                "val_acc":    round(val_acc,    4),
                "best_val_acc": round(best_val_acc, 4),
            }
            history.append(epoch_data)
            yield f"data: {json.dumps({'type': 'epoch', **epoch_data})}\n\n"

        # ── Save run record ──────────────────────────────────────────────────────
        duration = int(time.time() - start_time)
        run = {
            "id":           int(time.time()),
            "created_at":   time.strftime("%Y-%m-%dT%H:%M:%S"),
            "epochs":       req.epochs,
            "batch_size":   req.batch_size,
            "lr":           req.lr,
            "best_val_acc": round(best_val_acc, 4),
            "duration_seconds": duration,
            "dataset": {
                "cat":     cat_count,
                "not_cat": not_cat_count,
            },
            "history": history,
        }
        save_run(run)

        # Reload model into memory
        _load_model()

        yield f"data: {json.dumps({'type': 'done', 'best_val_acc': round(best_val_acc, 4), 'duration_seconds': duration})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/train/cancel")
def cancel_training():
    global _cancel_training
    _cancel_training = True
    return {"status": "ok", "message": "Cancel signal sent."}


@app.get("/train/runs")
def list_runs():
    return load_runs()


@app.delete("/data/clear")
def clear_data():
    """Delete all images from cat and not_cat folders."""
    import shutil
    for d in [CAT_DIR, NOT_CAT_DIR]:
        p = Path(d)
        if p.exists():
            shutil.rmtree(d)
            p.mkdir(parents=True, exist_ok=True)
    return {"status": "ok", "cat": 0, "not_cat": 0}


@app.post("/reset")
def reset_all():
    """Wipe model weights and training history. Images are kept."""
    if Path(MODEL_PATH).exists():
        Path(MODEL_PATH).unlink()
    Path(RUNS_FILE).write_text("[]")
    global _model, _model_ready
    with _model_lock:
        _model       = None
        _model_ready = False
    return {"status": "ok"}