# FundusNet — Retinal Disease Screening System

A production-grade Django web application for automated retinal disease screening using a deep learning ensemble. Classifies fundus photographs into **Healthy**, **Cataract**, **Glaucoma**, or **Retina Disease** with confidence scores, uncertainty quantification, and Grad-CAM explainability.

## Features

- **6-model ensemble**: EfficientNet-B0, ResNet50, SqueezeNet1.0, MobileNetV3-Small, ConvNeXt-Tiny, ViT-B/16
- **Test-time augmentation (TTA)**: 7 variants per image for robust predictions
- **Selective ensemble**: Filters outlier models when agreement is low
- **MC Dropout uncertainty**: Entropy-based refusal for low-confidence predictions
- **Grad-CAM heatmaps**: Visual explainability for every prediction
- **Fundus validation**: 5-signal heuristic rejects non-fundus images before inference
- **Image quality check**: Blur, brightness, contrast, saturation, edge scoring
- **LRU image cache**: 100-entry / 50 MB in-memory cache
- **Research-grade evaluation suite**: 5-fold CV, calibration metrics, ablation studies, statistical tests
- **Production security**: HSTS, CSP, rate limiting, django-axes brute-force protection

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Django 5.2, Python 3.9+ |
| ML | PyTorch 2.7, torchvision 0.22 |
| Image processing | OpenCV 4.12, Pillow 11.3 |
| Evaluation | scikit-learn, SciPy, Matplotlib, Seaborn |
| Auth throttling | django-axes |
| Database | SQLite (default), PostgreSQL-ready |
| Code quality | Black, isort, flake8 |

## Quick Start

```bash
git clone https://github.com/Mariakevin/FundusNet.git
cd FundusNet

python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
copy .env.example .env     # Edit as needed
python manage.py migrate
python manage.py runserver
```

Open http://127.0.0.1:8000

## Dataset

The `retina_dataset/` directory contains 597 fundus photographs across 4 classes:

| Class | Images |
|---|---|
| Healthy | 299 (50.1%) |
| Cataract | 100 (16.7%) |
| Glaucoma | 99 (16.6%) |
| Diabetic Retinopathy | 99 (16.6%) |

## Training

```bash
python train.py --models efficientnet resnet squeezenet --epochs 15
```

Supports 6 architectures with class-weighted loss, early stopping, and automatic checkpointing.

## Evaluation

```bash
python -m evaluation.evaluate --dataset retina_dataset --folds 5
python -m evaluation.baselines --dataset retina_dataset --folds 5
python -m evaluation.ablation --dataset retina_dataset --folds 5
```

The evaluation suite includes 5-fold stratified CV, ensemble strategy comparison, component ablation, uncertainty calibration, and publication-quality figure generation.

## Tests

```bash
python manage.py test retina_app
```

## Project Structure

```
FundusNet/
├── retina_app/             # Django application
│   ├── services/           # Inference, ensemble, Grad-CAM, uncertainty, preprocessing
│   ├── management/         # Custom management commands
│   ├── templates/          # UI templates (SPA)
│   └── tests/              # 15 test files
├── retina_project/         # Django project settings (dev/prod)
├── evaluation/             # Research-grade evaluation suite
├── docs/                   # Full documentation, API reference, paper draft
├── models/                 # Pretrained model checkpoints
├── retina_dataset/         # Fundus image dataset (4 classes)
├── train.py                # ML training script
└── manage.py               # Django entry point
```

## Documentation

Full documentation is in `docs/`:

- [API Reference](docs/API.md)
- [Dataset Details](docs/DATASET.md)
- [Developer Guide](docs/DEVELOPER.md)
- [Changelog](docs/CHANGELOG.md)
