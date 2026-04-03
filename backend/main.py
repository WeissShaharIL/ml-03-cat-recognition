import os
import json
import time
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_DIR   = os.getenv("MODEL_DIR",   "/app/model")
DATA_DIR    = os.getenv("DATA_DIR",    "/app/data")
CAT_DIR     = os.getenv("CAT_DIR",     "/app/data/cat")
NOT_CAT_DIR = os.getenv("NOT_CAT_DIR", "/app/data/not_cat")
RUNS_FILE   = os.getenv("RUNS_FILE",   "/app/data/runs.json")
DEPLOYED_FILE = Path(MODEL_DIR) / "deployed.txt"   # contains filename of active model

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
_model            = None
_model_ready      = False
_model_lock       = threading.Lock()
_cancel_training  = False
_deployed_model   = None   # filename of currently deployed model (e.g. "model_1743612345.pth")

# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    Path(CAT_DIR).mkdir(parents=True, exist_ok=True)
    Path(NOT_CAT_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    if not Path(RUNS_FILE).exists():
        Path(RUNS_FILE).write_text("[]")

    # Load deployed model if one exists
    if DEPLOYED_FILE.exists():
        deployed = DEPLOYED_FILE.read_text().strip()
        model_path = Path(MODEL_DIR) / deployed
        if model_path.exists():
            print(f"Loading deployed model: {deployed}")
            _load_model_from(str(model_path), deployed)
            return
    print("No deployed model found. Train and deploy via the dashboard.")

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


def get_deployed_filename() -> Optional[str]:
    if DEPLOYED_FILE.exists():
        name = DEPLOYED_FILE.read_text().strip()
        if (Path(MODEL_DIR) / name).exists():
            return name
    return None


def list_model_files() -> list[dict]:
    """Return all .pth files in MODEL_DIR with metadata from runs.json."""
    runs = {str(r.get("model_file", "")): r for r in load_runs()}
    deployed = get_deployed_filename()
    models = []
    for f in sorted(Path(MODEL_DIR).glob("model_*.pth"), reverse=True):
        run = runs.get(f.name, {})
        models.append({
            "filename":      f.name,
            "created_at":    run.get("created_at", ""),
            "best_val_acc":  run.get("best_val_acc"),
            "epochs":        run.get("epochs"),
            "batch_size":    run.get("batch_size"),
            "lr":            run.get("lr"),
            "duration_seconds": run.get("duration_seconds"),
            "dataset":       run.get("dataset", {}),
            "deployed":      f.name == deployed,
        })
    return models


def _load_model_from(path: str, filename: str):
    global _model, _model_ready, _deployed_model
    try:
        import torch
        import torchvision.models as models

        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()

        with _model_lock:
            _model          = model
            _model_ready    = True
            _deployed_model = filename
        print(f"Model loaded: {filename}")
    except Exception as e:
        print(f"Failed to load model {filename}: {e}")


def _preprocess_image(image_bytes: bytes):
    from PIL import Image
    import torchvision.transforms as transforms
    import io

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(img).unsqueeze(0)


# ── Schemas ────────────────────────────────────────────────────────────────────
class TrainRequest(BaseModel):
    epochs:     int   = DEFAULT_EPOCHS
    batch_size: int   = DEFAULT_BATCH_SIZE
    lr:         float = DEFAULT_LR

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":          "ok",
        "model_ready":     _model_ready,
        "deployed_model":  _deployed_model,
        "dataset": {
            "cat":     count_images(CAT_DIR),
            "not_cat": count_images(NOT_CAT_DIR),
        }
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not _model_ready:
        raise HTTPException(status_code=503, detail="No model deployed. Train and deploy first.")

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
        "prediction":    prediction,
        "confidence":    round(confidence, 4),
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
        try:
            full_dataset = ImageFolder(root=DATA_DIR, transform=train_transform)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': f'Failed to load dataset: {str(e)}'})}\n\n"
            return

        total      = len(full_dataset)
        val_size   = max(1, int(total * 0.2))
        train_size = total - val_size
        train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        val_ds.dataset   = ImageFolder(root=DATA_DIR, transform=val_transform)

        train_loader = DataLoader(train_ds, batch_size=req.batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=req.batch_size, shuffle=False, num_workers=0)

        yield f"data: {json.dumps({'type': 'status', 'text': f'Train: {train_size} images, Val: {val_size} images. Building model...'})}\n\n"

        # ── Model ────────────────────────────────────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        yield f"data: {json.dumps({'type': 'status', 'text': f'Using device: {str(device)}'})}\n\n"

        model = models.resnet18(weights="IMAGENET1K_V1")
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, 2)
        model    = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=req.lr)

        # ── Training loop ────────────────────────────────────────────────────────
        start_time   = time.time()
        best_val_acc = 0.0
        history      = []
        model_file   = f"model_{int(start_time)}.pth"
        model_path   = Path(MODEL_DIR) / model_file

        yield f"data: {json.dumps({'type': 'status', 'text': f'Starting training for {req.epochs} epochs...'})}\n\n"

        for epoch in range(req.epochs):
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

            # Save best weights for this run
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), str(model_path))

            epoch_data = {
                "epoch":        epoch + 1,
                "epochs":       req.epochs,
                "train_loss":   round(train_loss, 4),
                "train_acc":    round(train_acc,  4),
                "val_loss":     round(val_loss,   4),
                "val_acc":      round(val_acc,    4),
                "best_val_acc": round(best_val_acc, 4),
            }
            history.append(epoch_data)
            yield f"data: {json.dumps({'type': 'epoch', **epoch_data})}\n\n"

        # ── Save run record ──────────────────────────────────────────────────────
        duration = int(time.time() - start_time)
        run = {
            "id":               int(start_time),
            "model_file":       model_file,
            "created_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
            "epochs":           req.epochs,
            "batch_size":       req.batch_size,
            "lr":               req.lr,
            "best_val_acc":     round(best_val_acc, 4),
            "duration_seconds": duration,
            "dataset": {
                "cat":     cat_count,
                "not_cat": not_cat_count,
            },
            "history": history,
        }
        save_run(run)

        yield f"data: {json.dumps({'type': 'done', 'model_file': model_file, 'best_val_acc': round(best_val_acc, 4), 'duration_seconds': duration})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/train/cancel")
def cancel_training():
    global _cancel_training
    _cancel_training = True
    return {"status": "ok", "message": "Cancel signal sent."}


@app.get("/train/runs")
def list_runs():
    return load_runs()


@app.get("/models")
def list_models():
    """List all trained model files with metadata."""
    return list_model_files()


@app.post("/models/{filename}/deploy")
def deploy_model(filename: str):
    """Instantly swap the active model in memory."""
    model_path = Path(MODEL_DIR) / filename
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model file {filename} not found.")

    # Write deployed.txt
    DEPLOYED_FILE.write_text(filename)

    # Load into memory instantly
    _load_model_from(str(model_path), filename)

    return {"status": "ok", "deployed": filename}


@app.post("/reset")
def reset_all():
    """Wipe all model files, deployed pointer and training history. Images kept."""
    import shutil
    # Delete all model files
    for f in Path(MODEL_DIR).glob("model_*.pth"):
        f.unlink()
    if DEPLOYED_FILE.exists():
        DEPLOYED_FILE.unlink()
    Path(RUNS_FILE).write_text("[]")

    global _model, _model_ready, _deployed_model
    with _model_lock:
        _model          = None
        _model_ready    = False
        _deployed_model = None

    return {"status": "ok"}