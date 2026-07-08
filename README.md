```
                   ╔══════════════════════════════════════╗
                   ║         🔬  FundusNet   🧠           ║
                   ║  Retinal Disease Screening System    ║
                   ╚══════════════════════════════════════╝
                                    ▓▓▓▓▓▓▓▓▓▓
                                 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                                ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                                ▓▓▓▓    ▓▓▓▓    ▓▓▓▓
                                ▓▓▓▓    ▓▓▓▓    ▓▓▓▓
                                 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                                  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                                   ▓▓▓▓▓▓▓▓▓▓▓▓
                              ░░░░░░░░░░░░░░░░░░░░░
                           ░░░░░░░░░░░░░░░░░░░░░░░░░░░
                          ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
                          ░░░░░░              ░░░░░░░░░░
                          ░░░░░░    ░░░░░░    ░░░░░░░░░░
                          ░░░░░░    ░░░░░░    ░░░░░░░░░░
                           ░░░░░░░░░░░░░░░░░░░░░░░░░░░
                            ░░░░░░░░░░░░░░░░░░░░░░░░░
```

<p align="center">
  <b>A production-grade web application for automated retinal disease screening</b><br>
  <i>6-model ensemble · Uncertainty quantification · Grad-CAM explainability · Research-grade evaluation</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Django-5.2-green?logo=django&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-2.7-red?logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/OpenCV-4.12-blueviolet?logo=opencv&logoColor=white">
  <img src="https://img.shields.io/badge/Model_Ensemble-6_models-important">
  <img src="https://img.shields.io/badge/Explainability-GradCAM-success">
  <img src="https://img.shields.io/badge/License-Proprietary-lightgrey">
</p>

---

## 👁️ Overview

FundusNet turns a fundus photograph into a **multi-class screening decision** — Healthy, Cataract, Glaucoma, or Retina Disease — with calibrated confidence, uncertainty awareness, and visual heatmaps. Designed for clinical research, it combines **6 deep learning architectures** into a selective ensemble that refuses low-certainty predictions and explains every output.

### What makes it different

| | Single-model AI | FundusNet |
|---|---|---|
| **Architecture** | One model, one vote | 6-model selective ensemble |
| **Certainty** | Blind confidence | MC Dropout + entropy-based refusal |
| **Explainability** | None | Grad-CAM heatmaps |
| **Image validation** | Accepts anything | 5-signal fundus heuristic |
| **Evaluation** | Accuracy only | 5-fold CV + calibration + ablation |
| **Reproducibility** | Manual | Seeded pipelines, CI-ready |

---

## 🧠 Core Capabilities

**6-model ensemble** — EfficientNet-B0, ResNet50, SqueezeNet1.0, MobileNetV3-Small, ConvNeXt-Tiny, ViT-B/16

```mermaid
graph TD
    A[Fundus Photo] --> B[5-Signal Fundus Validation]
    B --> C[Quality Check]
    C --> D[Preprocessing]
    D --> E[6-Model Ensemble]
    
    E --> F1[EfficientNet-B0]
    E --> F2[ResNet50]
    E --> F3[SqueezeNet1.0]
    E --> F4[MobileNetV3]
    E --> F5[ConvNeXt-Tiny]
    E --> F6[ViT-B/16]
    
    F1 & F2 & F3 & F4 & F5 & F6 --> G{Agreement ≥ 0.5?}
    G -->|Yes| H[Weighted Ensemble]
    G -->|No| I[Selective Filtering]
    I --> H
    
    H --> J[MC Dropout<br/>(T=10 passes)]
    J --> K{Uncertainty < τ?}
    K -->|Yes| L[Prediction + Grad-CAM]
    K -->|No| M[Refuse: refer to specialist]
    
    L --> N[4-Class Output<br/>Confidence · Heatmap · Report]
```

**Uncertainty-aware refusal** — When MC Dropout entropy exceeds a threshold, the system refuses to classify and recommends manual review — a critical safety feature for clinical deployment.

**Grad-CAM explainability** — Heatmaps overlay on the original image highlight which regions drove the model's decision, providing visual accountability for every prediction.

**Fundus validation** — A 5-signal heuristic (color, circularity, edge density, green-channel variance, texture regularity) rejects non-fundus images before inference, preventing out-of-distribution errors.

---

## 🗂️ Project Structure

```
FundusNet/
├── retina_app/              # Django application
│   ├── services/             # Inference, ensemble, Grad-CAM, uncertainty
│   │   ├── inference.py      # Main orchestrator
│   │   ├── ensemble.py       # Selective ensemble + TTA
│   │   ├── gradcam.py        # Explainability heatmaps
│   │   ├── uncertainty.py    # MC Dropout quantification
│   │   ├── preprocessing.py  # CLAHE, ROI, quality checks
│   │   ├── fundus_validator.py  # 5-signal heuristic
│   │   └── model_manager.py  # Lazy-loaded singleton
│   ├── templates/            # SPA UI
│   ├── tests/                # 15 test files
│   └── management/           # Custom commands
├── retina_project/           # Django settings (dev/prod)
├── evaluation/               # 5-fold CV, ablation, statistics
│   ├── evaluate.py           # Main evaluation entry
│   ├── metrics.py            # ECE, MCE, Brier, AUROC
│   ├── statistics.py         # McNemar, DeLong, bootstrap CI
│   ├── ablation.py           # Component contribution study
│   └── figures.py            # Publication-quality plots
├── docs/                     # Full documentation suite
│   ├── API.md                # API reference
│   ├── PAPER_DRAFT.md        # Manuscript draft
│   ├── RELATED_WORK.md       # Literature survey
│   └── CHANGELOG.md          # Version history
└── train.py                  # Training script (6 architectures)
```

---

## 📊 Dataset

| Class | Images | Proportion |
|---|---|---|
| **Healthy** | 299 | 50.1% |
| **Cataract** | 100 | 16.7% |
| **Glaucoma** | 99 | 16.6% |
| **Diabetic Retinopathy** | 99 | 16.6% |

597 fundus photographs across 4 classes. Class imbalance addressed via weighted CrossEntropyLoss (`[1.0, 3.0, 3.0, 3.0]`). Evaluation uses 5-fold stratified cross-validation (seed 42).

---

## ⚡ Quick Start

```bash
# Clone and enter
git clone https://github.com/Mariakevin/FundusNet.git
cd FundusNet

# Environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Dependencies
pip install -r requirements.txt

# Setup
copy .env.example .env       # Edit as needed
python manage.py migrate
python manage.py runserver
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000)

### Training

```bash
python train.py --models efficientnet resnet squeezenet --epochs 15
```

Supports class-weighted loss, early stopping (patience=5), and automatic best-model checkpointing.

### Evaluation

```bash
python -m evaluation.evaluate --dataset retina_dataset --folds 5
python -m evaluation.baselines --dataset retina_dataset --folds 5
python -m evaluation.ablation --dataset retina_dataset --folds 5
```

### Tests

```bash
python manage.py test retina_app
```

---

## 🔧 Configuration

Settings live in `retina_project/settings/`:

| Environment | File | Key features |
|---|---|---|
| **Development** | `dev.py` | DEBUG=True, relaxed security, verbose logging |
| **Production** | `prod.py` | HSTS, CSP, forced HTTPS, secure cookies |
| **Shared** | `base.py` | Apps, middleware, database, auth, logging |

All ML and application constants are centralized in `retina_app/constants.py` (147 lines) — model weights, thresholds, cache limits, preprocessing settings. Single source of truth, zero duplication.

---

## 🧪 Research-Grade Evaluation Suite

| Module | What it measures |
|---|---|
| `metrics.py` | Accuracy, F1, ECE, MCE, Brier score, per-class metrics, AUROC |
| `statistics.py` | McNemar's test, paired t-test, DeLong AUC test, bootstrap CI, Bonferroni correction |
| `baselines.py` | 5 ensemble strategies + published literature comparison |
| `ablation.py` | Contribution of each system component |
| `uncertainty_eval.py` | MC Dropout vs. softmax vs. ensemble disagreement — accuracy-refusal tradeoffs |
| `figures.py` | Confusion matrices, ROC curves, reliability diagrams, Grad-CAM grids |

---

## 🔒 Security

- **Throttled auth**: django-axes brute-force protection (configurable lockout)
- **Production hardening**: HSTS (1 year), CSP, X-Frame-Options DENY, Referrer Policy
- **File validation**: MIME type check, path traversal protection, 10 MB size limit
- **Soft delete**: Prediction records use logical deletion

---

## 📚 Documentation

| Document | Contents |
|---|---|
| [API Reference](docs/API.md) | All views, endpoints, forms, and templates |
| [Dataset Details](docs/DATASET.md) | Preprocessing, statistics, ethical considerations |
| [Developer Guide](docs/DEVELOPER.md) | Workflow, testing, contribution guide |
| [Paper Draft](docs/PAPER_DRAFT.md) | Full manuscript with methodology and results |
| [Changelog](docs/CHANGELOG.md) | Version history |

---

<p align="center">
  <i>Built with Django, PyTorch, and OpenCV</i><br>
  <b>FundusNet</b> — interpretable, uncertainty-aware retinal disease screening
</p>
