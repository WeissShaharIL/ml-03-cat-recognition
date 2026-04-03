# ml-03-cat-recognition

A local image classifier that learns to recognize cats.
Drop an image into the UI, get a prediction and confidence score — powered by a fine-tuned ResNet18.

This is a **transfer learning** project: a model pretrained on ImageNet (1.2M images, 1000 classes)
is retrained on the last layer to distinguish cats from non-cats.
Training happens locally inside Docker. Inference is instant.

---

## How it works

```
Pretrained ResNet18 (ImageNet weights)
    → Freeze all layers except the final FC layer
    → Replace FC: 1000 classes → 2 classes (cat / not_cat)
    → Fine-tune on your image dataset
    → Save best weights per run as model_<timestamp>.pth

User drops image into classifier UI
    → Resize to 224×224, normalize (ImageNet mean/std)
    → Forward pass through fine-tuned ResNet18
    → Softmax → probabilities
    → Return: cat 82.4% / not_cat 17.6%
```

The model weights are **frozen after training**. Inference is a single forward pass.
Multiple trained models can coexist — you deploy any one instantly from the dashboard.

---

## Architecture

```
Classifier UI (React :3000)     Dashboard UI (React :3001)
        │                                │
        │  POST /predict                 │  POST /train (SSE)
        │  (image upload)                │  POST /models/{file}/deploy
        ▼                                ▼
              FastAPI Backend (:8000)
                      │
              ResNet18 (PyTorch)
              loaded from deployed model
```

| Service | Port | Role |
|---|---|---|
| Classifier UI | 3000 | Drop image → get prediction + confidence |
| Dashboard UI | 3001 | Train models, compare runs, deploy |
| FastAPI Backend | 8000 | Inference, training pipeline, model versioning |

---

## Classifier UI

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

## Dashboard UI

3-column layout:

```
Left sidebar          Center                      Right sidebar
─────────────         ──────────────────────      ─────────────
Service health        Train Model                 Trained Models
Dataset counts          Epochs / Batch / LR         model_xyz.pth
  🐱 400 imgs           ▶ Train  ✕ Cancel           ● live  92.1%
  ✗  400 imgs         Live training log             🚀 Deploy
                        per-epoch metrics
Link to classifier    Training History            model_abc.pth
                        expandable per-epoch        latest  89.4%
                        loss/accuracy table         🚀 Deploy
```

**Top bar** shows which model is currently serving the classifier:
`Serving: model_1743612345.pth` or `No model deployed`

---

## Getting Started

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Node.js 20+](https://github.com/coreybutler/nvm-windows)

### 1. Add training images

Create two folders and populate them:

```
data/
  cat/        ← cat images (.jpg / .png)
  not_cat/    ← non-cat images
```

Recommended: ~400 images per class, balanced. More is better.

Good free sources:
- [Microsoft Cats vs Dogs](https://www.microsoft.com/en-us/download/details.aspx?id=54746) — direct download, no account
- [Kaggle Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) — variety of non-cat classes
- [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) — buildings/nature scenes, no animals

### 2. Start the app

```powershell
.\build.ps1
```

### 3. Train a model

1. Open http://localhost:3001 (dashboard)
2. Check dataset counts in the left sidebar
3. Set epochs/batch/lr and click **▶ Train Model**
4. Watch live loss and accuracy per epoch
5. When done, click **🚀 Deploy** on the model in the right sidebar
6. Open http://localhost:3000 and drop in an image

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

## Model versioning

Each training run saves a separate file:

```
model/
  model_1743612345.pth    ← run 1
  model_1743615678.pth    ← run 2  ← deployed
  deployed.txt            ← contains filename of active model
```

- Multiple models coexist on disk
- Deploying a model swaps it **instantly in memory** — no restart needed
- The dashboard shows accuracy, dataset size, training time per model
- Reset wipes all models and history but keeps your images

---

## Training details

| Parameter | Default | Notes |
|---|---|---|
| Base model | ResNet18 | Pretrained on ImageNet |
| Fine-tuning | Final FC layer only | All other layers frozen |
| Input size | 224 × 224 px | ImageNet standard |
| Normalization | ImageNet mean/std | Required for pretrained weights |
| Classes | `cat`, `not_cat` | Determined by folder names |
| Optimizer | Adam | |
| Loss | CrossEntropyLoss | |
| Train/val split | 80/20 | Random split each run |
| Augmentation | HorizontalFlip + ColorJitter | Training only |

**Training time estimates (CPU):**

| Images | Epochs | Time |
|---|---|---|
| 400+400 | 5 | ~8 min |
| 800+800 | 5 | ~15 min |
| 400+400 | 10 | ~15 min |

---

## Project structure

```
ml-03-cat-recognition/
├── .env                        ← training defaults
├── .gitignore
├── docker-compose.yml
├── build.ps1                   ← builds Docker images and starts services
├── README.md
├── data/
│   ├── cat/                    ← cat training images (gitignored)
│   ├── not_cat/                ← non-cat training images (gitignored)
│   └── runs.json               ← training run history
├── model/
│   ├── model_<timestamp>.pth   ← one file per training run (gitignored)
│   └── deployed.txt            ← filename of active model
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py                 ← FastAPI: train, predict, deploy, versioning
├── frontend-classifier/
│   ├── Dockerfile
│   ├── nginx.conf
│   └── src/App.jsx             ← drag & drop classifier UI
└── frontend-dash/
    ├── Dockerfile
    ├── nginx.conf
    └── src/App.jsx             ← dashboard: train, models, deploy
```

---

## Backend API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status, deployed model, dataset counts |
| POST | `/predict` | Upload image → cat/not_cat probabilities |
| POST | `/train` | Fine-tune ResNet18 — SSE stream of metrics |
| POST | `/train/cancel` | Cancel in-progress training |
| GET | `/train/runs` | Training history (from runs.json) |
| GET | `/models` | List all trained model files with metadata |
| POST | `/models/{filename}/deploy` | Instantly swap active model in memory |
| POST | `/reset` | Wipe models + history (images kept) |

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

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + Vite |
| Backend | FastAPI (Python) |
| ML framework | PyTorch + torchvision |
| Base model | ResNet18 (pretrained ImageNet) |
| Serving | Nginx + Docker Compose |

---

## Roadmap

- [x] Phase 1 — Project skeleton: all services wired, Docker Compose
- [x] Phase 2 — Training pipeline: ResNet18 fine-tune, SSE progress stream, cancel
- [x] Phase 3 — Classifier UI: drag & drop, confidence bars
- [x] Phase 4 — Model versioning: save per run, deploy instantly, dashboard

---

## Ideas for next steps

### Quick wins
- **Confidence threshold** — show "Not sure 🤔" when confidence is below ~60% instead of forcing a prediction. Makes the classifier more honest.
- **Test set panel** — keep a small `data/test/` folder of labeled images. Dashboard runs the deployed model against them automatically and shows pass/fail per image.
- **Model comparison** — when deploying a new model, auto-run both against the test set and show a before/after accuracy diff.

### Bigger improvements
- **Unfreeze more layers** — currently only the final FC layer trains. Unfreezing the last ResNet block (layer4) would improve accuracy at the cost of longer training time.
- **Data augmentation controls** — expose flip, rotation, brightness, contrast as dashboard settings so you can experiment without touching code.
- **Confusion matrix** — after each training run, show true positives / false positives / false negatives on the validation set. Tells you exactly what the model gets wrong.
- **Class activation maps (CAM)** — visualize which pixels the model looked at when making a prediction. Helps debug bad predictions.

### Phase 5 — Multi-class
- Extend beyond cat/not_cat to recognize multiple animal species
- Add more class folders to `data/` and the model adapts automatically (ImageFolder picks up any folder structure)
- Dashboard would show per-class accuracy breakdown

### Phase 6 — Object detection
- Move from classification (is there a cat?) to detection (where is the cat?)
- Replace ResNet18 with a YOLO or SSD model
- Draw bounding boxes on the image in the classifier UI
