"""Custom exceptions for RetinaAI application.
Provides specific error types for better error handling and debugging.
"""


class RetinaAIError(Exception):
    """Base exception for all RetinaAI errors."""

    def __init__(self, message: str, code: str = "UNKNOWN_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)


class ImageValidationError(RetinaAIError):
    """Raised when image validation fails."""

    def __init__(self, message: str):
        super().__init__(message, code="IMAGE_VALIDATION_ERROR")


class ImageCorruptError(RetinaAIError):
    """Raised when image file is corrupt or unreadable."""

    def __init__(self, message: str = "Image file is corrupt or unreadable"):
        super().__init__(message, code="IMAGE_CORRUPT_ERROR")


class ImageSizeError(RetinaAIError):
    """Raised when image size exceeds limits."""

    def __init__(self, message: str):
        super().__init__(message, code="IMAGE_SIZE_ERROR")


class ImageDimensionError(RetinaAIError):
    """Raised when image dimensions are invalid."""

    def __init__(self, message: str):
        super().__init__(message, code="IMAGE_DIMENSION_ERROR")


class ModelLoadError(RetinaAIError):
    """Raised when model fails to load."""

    def __init__(self, message: str):
        super().__init__(message, code="MODEL_LOAD_ERROR")


class InferenceError(RetinaAIError):
    """Raised when model inference fails."""

    def __init__(self, message: str):
        super().__init__(message, code="INFERENCE_ERROR")


class PreprocessingError(RetinaAIError):
    """Raised when image preprocessing fails."""

    def __init__(self, message: str):
        super().__init__(message, code="PREPROCESSING_ERROR")


class NotAFundusImageError(RetinaAIError):
    """Raised when uploaded image is not a retinal fundus photograph."""

    def __init__(self, message: str = "Image does not appear to be a retinal fundus photograph"):
        super().__init__(message, code="NOT_A_FUNDUS_IMAGE")
