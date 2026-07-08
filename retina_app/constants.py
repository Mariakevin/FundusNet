"""Centralized constants for FundusNet application.
Single source of truth for validation limits, categories, and model configuration.
"""

# --- Image Validation ---
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp", "image/tiff"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_IMAGE_DIMENSION = 4096
MIN_IMAGE_DIMENSION = 64

# --- Classification Categories ---
# Must match training categories in train.py exactly
CATEGORIES = ["Healthy", "Cataract", "Glaucoma", "Retina Disease"]
CATEGORY_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}

# --- Model Configuration ---
MODEL_LIST = ["squeezenet", "efficientnet", "resnet", "mobilenet", "convnext", "vit"]

MODEL_WEIGHTS = {
    "squeezenet": 0.15,
    "efficientnet": 0.25,
    "resnet": 0.20,
    "mobilenet": 0.10,
    "convnext": 0.15,
    "vit": 0.15,
}

# Per-class performance weights for dynamic ensemble balancing
CLASS_PERFORMANCE_WEIGHTS = {
    "Healthy": {
        "squeezenet": 0.15,
        "efficientnet": 0.25,
        "resnet": 0.20,
        "mobilenet": 0.10,
        "convnext": 0.15,
        "vit": 0.15,
    },
    "Cataract": {
        "squeezenet": 0.10,
        "efficientnet": 0.25,
        "resnet": 0.20,
        "mobilenet": 0.10,
        "convnext": 0.20,
        "vit": 0.15,
    },
    "Glaucoma": {
        "squeezenet": 0.10,
        "efficientnet": 0.20,
        "resnet": 0.25,
        "mobilenet": 0.10,
        "convnext": 0.15,
        "vit": 0.20,
    },
    "Retina Disease": {
        "squeezenet": 0.15,
        "efficientnet": 0.25,
        "resnet": 0.20,
        "mobilenet": 0.10,
        "convnext": 0.15,
        "vit": 0.15,
    },
}

ENSEMBLE_MIN_MODELS = 2
MAX_WORKERS = 4

# --- Model Health Monitoring ---
MODEL_HEALTH_WINDOW = 100  # Number of recent predictions to track per model
MODEL_HEALTH_MIN_ACCURACY = 0.3  # Minimum accuracy before model is flagged unhealthy

# --- Confidence Thresholds ---
CONFIDENCE_THRESHOLD_REFUSE = 0.35  # Refuse prediction if confidence is below this (OOD catch)
CONFIDENCE_THRESHOLD_LOW = 0.5
CONFIDENCE_THRESHOLD_HIGH = 0.7

# --- Caching ---
MAX_CACHE_SIZE = 100
MAX_CACHE_MEMORY_MB = 50

# --- TTA ---
TTA_AGGREGATION_METHOD = "geometric"
TEMPERATURE_SCALING = 1.0

# --- MC Dropout Uncertainty Quantification ---
MC_DROPOUT_PASSES = 10  # Number of stochastic forward passes for uncertainty estimation
UNCERTAINTY_THRESHOLD = 0.3  # Refuse classification if entropy exceeds this
ENABLE_MC_DROPOUT = False  # Off by default (opt-in)
UNCERTAINTY_REFUSAL_MESSAGE = (
    "The model is uncertain about this classification. Please consult an ophthalmologist for manual review."
)

# --- Grad-CAM Explainability ---
GRADCAM_MODEL = "efficientnet"  # Primary model for Grad-CAM heatmap
GRADCAM_ALPHA = 0.5  # Blending factor (0 = original only, 1 = heatmap only)
GRADCAM_COLORMAP = "jet"  # OpenCV colormap for heatmap

# --- Fundus Image Validation ---
FUNDUS_VALIDATION_ENABLED = True
FUNDUS_VALIDATION_THRESHOLD = 0.4  # Minimum combined score to accept as fundus
FUNDUS_COLOR_MIN_RATIO = 0.30  # Minimum % of pixels in fundus color range (red-orange)
FUNDUS_CIRCULARITY_MIN = 0.3  # Minimum circularity score for bright region
FUNDUS_AREA_MIN_RATIO = 0.20  # Minimum bright region area ratio
FUNDUS_AREA_MAX_RATIO = 0.90  # Maximum bright region area ratio
FUNDUS_EDGE_MIN_RATIO = 0.01  # Minimum edge pixel ratio
FUNDUS_EDGE_MAX_RATIO = 0.25  # Maximum edge pixel ratio (text >0.25)
FUNDUS_GREEN_CH_MIN_STD = 15.0  # Minimum green channel std dev

# --- Adaptive CLAHE ---
ADAPTIVE_CLAHE_ENABLED = False  # Disabled: not used during training, causes out-of-distribution shift
ADAPTIVE_CLAHE_DARK_CLIP = 3.0
ADAPTIVE_CLAHE_NORMAL_CLIP = 2.0
ADAPTIVE_CLAHE_BRIGHT_CLIP = 1.5
ADAPTIVE_CLAHE_DARK_THRESHOLD = 80
ADAPTIVE_CLAHE_BRIGHT_THRESHOLD = 180
ADAPTIVE_CLAHE_DARK_TILE = (8, 8)
ADAPTIVE_CLAHE_NORMAL_TILE = (8, 8)
ADAPTIVE_CLAHE_BRIGHT_TILE = (16, 16)

# --- Noise Reduction ---
NOISE_REDUCTION_ENABLED = False  # Disabled: not used during training, causes out-of-distribution shift
NOISE_REDUCTION_STRENGTH = 10
NOISE_REDUCTION_SPECULAR_ENABLED = True
NOISE_REDUCTION_SPECULAR_KERNEL = 5

# --- Color Constancy ---
COLOR_CONSTANCY_ENABLED = False  # Disabled: not used during training, causes out-of-distribution shift
COLOR_CONSTANCY_METHOD = "gray_world"
COLOR_CONSTANCY_WHITE_PATCH_PERCENTILE = 99

# --- Preprocessing Visualization ---
PREPROCESSING_VIZ_ENABLED = False  # Disabled along with preprocessing steps
PREPROCESSING_VIZ_PANELS = ["original", "clahe", "denoised", "color_corrected"]

# --- Learned Fundus Validator ---
FUNDUS_LEARNED_VALIDATOR_ENABLED = False  # Opt-in: learned binary classifier for fundus detection
FUNDUS_LEARNED_MODEL_PATH = "models/fundus_classifier.pth"
FUNDUS_LEARNED_THRESHOLD = 0.5  # Probability threshold for fundus classification

# --- Dynamic Weight Learning ---
ENSEMBLE_LEARNED_WEIGHTS_ENABLED = False  # Opt-in: use optimization-learned weights
ENSEMBLE_LEARNED_WEIGHTS_PATH = "models/learned_ensemble_weights.json"

# --- Evaluation ---
EVALUATION_N_FOLDS = 5
EVALUATION_SEED = 42
