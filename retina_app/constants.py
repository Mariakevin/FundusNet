"""Centralized constants for FundusNet application.
Single source of truth for validation limits, categories, and model configuration.
"""

# --- Image Validation ---
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp", "image/tiff"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_IMAGE_DIMENSION = 4096
MIN_IMAGE_DIMENSION = 64
FUNDUS_MIN_TOP1_TOP2_RATIO = 1.8

# --- Classification Categories ---
CATEGORIES = ["Healthy", "Cataract", "Glaucoma", "Retina Disease"]

# --- Model Configuration ---
# 5-model ensemble: dual-branch (CNN + Transformer) + hybrid architectures
# Based on 2025-2026 research: SwinTiny + hybrid models dominate multi-label screening
MODEL_LIST = ["swin", "maxvit", "convnext_v2", "efficientnet_v2", "deit"]

MODEL_WEIGHTS = {
    "swin": 0.25,  # Best single model on RFMiD benchmark (2026)
    "maxvit": 0.25,  # Hybrid CNN-Transformer with multi-axis attention
    "convnext_v2": 0.20,  # Strong CNN baseline
    "efficientnet_v2": 0.15,  # Efficient deployment
    "deit": 0.15,  # Vision Transformer baseline
}

CLASS_PERFORMANCE_WEIGHTS = {
    "Healthy": {"swin": 0.25, "maxvit": 0.25, "convnext_v2": 0.25, "efficientnet_v2": 0.15, "deit": 0.10},
    "Cataract": {"swin": 0.25, "maxvit": 0.25, "convnext_v2": 0.20, "efficientnet_v2": 0.25, "deit": 0.05},
    "Glaucoma": {"swin": 0.30, "maxvit": 0.25, "convnext_v2": 0.20, "efficientnet_v2": 0.15, "deit": 0.10},
    "Retina Disease": {"swin": 0.25, "maxvit": 0.30, "convnext_v2": 0.20, "efficientnet_v2": 0.15, "deit": 0.10},
}

ENSEMBLE_MIN_MODELS = 2
MAX_WORKERS = 4

# --- Learnable Fusion (Dynamic Weighting) ---
LEARNABLE_FUSION_ENABLED = True
LEARNABLE_FUSION_MLP_HIDDEN = 64
LEARNABLE_FUSION_DROPOUT = 0.3

# --- Stacking Meta-Learner ---
STACKING_ENABLED = True
STACKING_META_LEARNER = "logistic"

# --- Model Health Monitoring ---
MODEL_HEALTH_WINDOW = 100
MODEL_HEALTH_MIN_ACCURACY = 0.3

# --- Confidence Thresholds ---
CONFIDENCE_THRESHOLD_REFUSE = 0.50
CONFIDENCE_THRESHOLD_LOW = 0.5
CONFIDENCE_THRESHOLD_HIGH = 0.7
OOD_ENTROPY_THRESHOLD = 0.70

# --- Caching ---
MAX_CACHE_SIZE = 100
MAX_CACHE_MEMORY_MB = 50

# --- TTA ---
TTA_AGGREGATION_METHOD = "geometric"
TEMPERATURE_SCALING = 1.0

# --- Uncertainty-Aware TTA (BayTTA-inspired) ---
UNCERTAINTY_AWARE_TTA_ENABLED = True
TTA_MIN_VARIANCE_THRESHOLD = 0.01

# --- MC Dropout Uncertainty ---
MC_DROPOUT_PASSES = 10
UNCERTAINTY_THRESHOLD = 0.3
ENABLE_MC_DROPOUT = False
UNCERTAINTY_REFUSAL_MESSAGE = (
    "The model is uncertain about this classification. Please consult an ophthalmologist for manual review."
)

# --- Disease Co-occurrence Matrix ---
DISEASE_CO_OCCURRENCE_ENABLED = True
DISEASE_CO_OCCURRENCE_MATRIX = {
    # [Healthy, Cataract, Glaucoma, Retina Disease]
    "Healthy": [1.0, 0.05, 0.08, 0.10],
    "Cataract": [0.05, 1.0, 0.12, 0.15],
    "Glaucoma": [0.08, 0.12, 1.0, 0.25],
    "Retina Disease": [0.10, 0.15, 0.25, 1.0],
}

# --- Long-Tail Aware Learning ---
LONG_TAIL_LOSS_ENABLED = True
LONG_TAIL_LOSS_BETA = 0.9999
LONG_TAIL_EFFECTIVE_NUM_ALPHA = 1.0

# --- Adapter Modules (RetExpert-inspired) ---
ADAPTER_ENABLED = True
ADAPTER_DIMENSION = 64
ADAPTER_DROPOUT = 0.1

# --- Grad-CAM Explainability ---
GRADCAM_MODEL = "swin"
GRADCAM_ALPHA = 0.5
GRADCAM_COLORMAP = "jet"

# --- Fundus Image Validation ---
FUNDUS_VALIDATION_ENABLED = True
FUNDUS_VALIDATION_THRESHOLD = 0.60
FUNDUS_COLOR_MIN_RATIO = 0.35
FUNDUS_CIRCULARITY_MIN = 0.3
FUNDUS_AREA_MIN_RATIO = 0.20
FUNDUS_AREA_MAX_RATIO = 0.90
FUNDUS_EDGE_MIN_RATIO = 0.01
FUNDUS_EDGE_MAX_RATIO = 0.25
FUNDUS_GREEN_CH_MIN_STD = 15.0
