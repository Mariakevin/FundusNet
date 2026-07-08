# FundusNet Dataset

## Source

The dataset consists of retinal fundus images collected for multi-class retinal disease screening. Images are organized into four diagnostic categories.

## Statistics

| Class | Folder | Images | Percentage |
|-------|--------|--------|------------|
| Healthy | `1_normal/` | 299 | 50.1% |
| Cataract | `2_cataract/` | 100 | 16.7% |
| Glaucoma | `3_glaucoma/` | 99 | 16.6% |
| Diabetic Retinopathy | `4_Diabetic_Retinopathy/` | 99 | 16.6% |
| **Total** | | **597** | **100%** |

### Class Imbalance

The dataset exhibits a ~3:1 imbalance between the Healthy class and each disease class. This is addressed during training via:

- **Class-weighted cross-entropy loss**: weights `[1.0, 3.0, 3.0, 3.0]` inversely proportional to class frequency
- **Stratified cross-validation**: ensures each fold preserves the class distribution
- **Data augmentation**: random crops, flips, rotations, and color jitter to increase effective sample size for minority classes

## Image Properties

- **Format**: PNG
- **Color space**: RGB
- **Resolution**: Variable (resized to 224x224 for model input)
- **Content**: Retinal fundus photographs showing the optic disc, macula, and retinal vasculature

## Data Splits

| Split | Ratio | Seed | Purpose |
|-------|-------|------|---------|
| Training | 80% | 42 | Model parameter optimization |
| Validation | 20% | 42 | Hyperparameter selection and early stopping |

For evaluation, **5-fold stratified cross-validation** is used (seed=42) to obtain robust performance estimates with confidence intervals.

## Preprocessing Pipeline

1. **Resize**: 256x256 (training) or 224x224 (inference)
2. **Random crop**: 224x224 (training only)
3. **Normalization**: ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`
4. **Optional CLAHE**: Contrast-limited adaptive histogram equalization (disabled by default)
5. **Optional noise reduction**: Non-local means denoising (disabled by default)
6. **Optional color constancy**: Gray-world algorithm (disabled by default)

## Data Augmentation (Training)

| Transform | Parameters | Rationale |
|-----------|------------|-----------|
| Random horizontal flip | p=0.5 | Fundus images are symmetric left-right |
| Random rotation | ±15 degrees | Optic disc position varies across patients |
| Color jitter | brightness/contrast/saturation=0.2 | Simulates different camera settings and lighting |
| Random crop | 224x224 from 256x256 | Invariance to spatial position |

## Ethical Considerations

- All images are de-identified (no patient metadata included)
- Dataset is used for research purposes only
- No demographic information (age, sex, ethnicity) is available
- No institutional review board (IRB) approval documentation is included with the dataset

## Usage in This Project

The dataset is used for:

1. **Training**: 6 model architectures (EfficientNet-B0, ResNet50, SqueezeNet1.0, MobileNetV3-Small, ConvNeXt-Tiny, ViT-B/16) with transfer learning from ImageNet
2. **Evaluation**: 5-fold stratified cross-validation with per-class metrics, calibration analysis, and ablation studies
3. **Baselines**: Comparison of ensemble strategies (single model, averaging, weighted, selective, MC Dropout refusal)

## Limitations

- Small dataset size (597 images) — results should be interpreted with caution
- No external validation on independent datasets
- No demographic stratification analysis
- Single-center data (if applicable) — generalizability unknown
- 3:1 class imbalance may bias toward the Healthy class
