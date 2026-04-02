# ml-03-cat-recognition

A local image classifier that learns to recognize cats.
Drop an image, get a prediction and confidence score — built on a fine-tuned ResNet18.

This is a transfer learning project: a pretrained model that already understands visual features
(edges, textures, shapes) is retrained on the last layer to distinguish cats from non-cats.
Training happens once, locally. Inference is fast.

---

## How it works

```
Pretrained ResNet18 (ImageNet weights)
    → Replace final layer (1000 classes → 2 classes: cat / not cat)
    → Fine-tune on Cats vs Dogs dataset
    → Save best model weights

User drops image
    → Resize + normalize (match training preprocessing)
    → Forward pass through fine-tuned ResNet18
    → Softmax → probabilities
    → Return: cat 82.4% / not cat 17.6%
```

The model weights are **frozen after training**. Inference is purely a forward pass.

---

## Architecture

```
Web UI (React :3000)
        │
        │  POST /predict  (image upload)
        ▼
FastAPI Backend (:8000)
        │
        │  loads model weights once on startup
        ▼
ResNet18 (PyTorch) — fine-tuned
        │
        ▼
{ cat: 0.824, not_cat: 0.176 }
```

| Service | Role |
|---|---|
| Web UI | Drag & drop image, show prediction + confidence |
| FastAPI Backend | Image preprocessing + model inference |
| Training script | Fine-tunes ResNet18, saves best weights |

---

## UI

```
┌─────────────────────────────────────┐
│                                     │
│     Drop an image here              │
│     or click to upload              │
│                                     │
├─────────────────────────────────────┤
│  🐱 Cat         82.4%  ████████░░   │
│  ✗  Not a cat   17.6%  ███░░░░░░░   │
│                                     │
│  Prediction: Cat  (82.4% confident) │
└─────────────────────────────────────┘
```

---

## Getting Started

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Node.js 20+](https://github.com/coreybutler/nvm-windows)
- Python 3.11+ (for running the training script locally)

### 1. Train the model

```powershell
cd training
pip install -r requirements.txt
python train.py
```

This will:
- Download the Cats vs Dogs dataset (~800MB)
- Fine-tune ResNet18 for a few epochs
- Save the best weights to `backend/model/resnet18_cats.pth`

Training time estimates:

| Hardware | Dataset size | Time |
|---|---|---|
| CPU | 2,000 images (subset) | ~10 min |
| CPU | 25,000 images (full) | ~90 min |
| GPU | 25,000 images (full) | ~5 min |

### 2. Start the app

```powershell
.\build.ps1
```

### Daily use

```powershell
# Start
docker compose up

# Stop
docker compose down

# Rebuild after code changes
.\build.ps1
```

---

## Project structure

```
ml-03-cat-recognition/
├── .env                        ← config (model path, threshold)
├── docker-compose.yml
├── build.ps1                   ← builds React frontend, starts Docker
├── README.md
├── training/
│   ├── requirements.txt        ← torch, torchvision, etc.
│   ├── train.py                ← download dataset, fine-tune, save weights
│   └── evaluate.py             ← test accuracy on held-out set
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                 ← FastAPI: /predict endpoint
│   └── model/
│       └── resnet18_cats.pth   ← saved model weights (gitignored)
└── frontend/
    ├── Dockerfile
    ├── nginx.conf
    └── src/
        └── App.jsx             ← drag & drop UI + confidence bars
```

---

## Backend API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service status + model loaded check |
| POST | `/predict` | Upload image → returns cat/not-cat probabilities |

### `/predict` request

```
POST /predict
Content-Type: multipart/form-data

file: <image file>
```

### `/predict` response

```json
{
  "prediction": "cat",
  "confidence": 0.824,
  "probabilities": {
    "cat": 0.824,
    "not_cat": 0.176
  }
}
```

---

## Model details

| Parameter | Value |
|---|---|
| Base model | ResNet18 (pretrained on ImageNet) |
| Fine-tuning | Last fully connected layer replaced + retrained |
| Input size | 224 × 224 px |
| Normalization | ImageNet mean/std |
| Classes | `cat`, `not_cat` |
| Dataset | Kaggle Cats vs Dogs |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + Vite |
| Backend | FastAPI (Python) |
| ML framework | PyTorch + torchvision |
| Base model | ResNet18 |
| Serving | Nginx + Docker Compose |

---

## Roadmap

- [ ] Phase 1 — Training pipeline: fine-tune ResNet18, save weights, plot loss/accuracy curves
- [ ] Phase 2 — Inference API: FastAPI `/predict` endpoint with image preprocessing
- [ ] Phase 3 — UI: drag & drop image, confidence bars, prediction label
- [ ] Phase 4 — Dashboard: training metrics, model info, sample predictions
