# Changelog

All notable changes to FundusNet are documented here.

---

## [1.0.0] - 2026-05-10

### Added
- **Complete Django Application**
  - User authentication (login, register, logout)
  - Image upload and classification
  - Prediction history with filtering and pagination
  - Batch processing for multiple images
  - Async job processing for long-running inference

- **ML Inference System**
  - Ensemble of 4 models (SqueezeNet, EfficientNet, ResNet, MobileNet)
  - Test-Time Augmentation (TTA) with 8 augmentations
  - CLAHE preprocessing for fundus images
  - ROI detection for circular fundus region
  - Image quality assessment
  - Confidence thresholds with warnings
  - Result caching (LRU, 50MB limit)

- **Custom Exception Handling**
  - InferenceError, ImageValidationError, ImageCorruptError
  - ImageSizeError, ImageDimensionError, ModelLoadError
  - PreprocessingError, PredictionTimeoutError

- **Production Security**
  - Rate limiting (10 req/min)
  - HSTS, CSP, Referrer Policy headers
  - Protected media file serving
  - Path traversal protection

- **Maximalist UI Design**
  - Vibrant color scheme (#E91E63 magenta, #9C27B0 purple)
  - Glowing effects and animations
  - Dark slate background (#1E2A3A)
  - Responsive design

### Changed
- Binary classification (Normal/Diseased) from multi-class
- Soft-delete for prediction records
- Improved model loading with compatibility layer
- Optimized database indexes for queries

### Removed
- Multi-class categories (DR, Glaucoma, etc.)
- Dead code from previous iterations

---

## [Pre-1.0.0] - Development Versions

### v0.9.x
- Initial Django project setup
- Basic model architecture
- First ML inference implementation

### v0.8.x
- User authentication system
- Database models and migrations

### v0.7.x
- Template design and styling
- Frontend JavaScript implementation

---

## Future Considerations

- [ ] Add more model architectures
- [ ] Implement true multi-class detection
- [ ] Add API for third-party integration
- [ ] Implement audit logging
- [ ] Add support for more image formats
- [ ] Mobile application companion

---

## Deprecation Notes

- Multi-class classification deprecated in favor of binary (Normal/Diseased)
- Old model file formats no longer supported
- Legacy URL patterns removed

---

## Migration Notes

### From v0.x to v1.0

1. Run database migrations:
   ```bash
   python manage.py migrate
   ```

2. Update model files (if using custom weights):
   - Ensure binary classification format
   - Update model paths in inference.py if needed

3. Update environment variables for production:
   ```bash
   DEBUG=False
   SECRET_KEY=<new-secret>
   ALLOWED_HOSTS=yourdomain.com
   ```

---

## Known Issues

- First inference request slower (model loading)
- Large batch processing may timeout (10 image limit)
- Limited to CPU inference (GPU not yet supported)

---

## Contact

For support and inquiries: development team
