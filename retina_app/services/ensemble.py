"""Ensemble prediction logic — upgraded with stacking meta-learner and uncertainty-aware TTA.

Upgrades:
- Stacking meta-learner (learns optimal non-linear combination of model predictions)
- Per-class dynamic weighted averaging (preserved)
- Selective ensemble with adaptive thresholding
- Model disagreement detection
- Uncertainty-aware TTA aggregation (BayTTA-inspired)
- Disease co-occurrence matrix for medical prior knowledge
- Learnable fusion weights (Res101-MViT-Ens inspired)
"""

import logging
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from retina_app.constants import (
    CATEGORIES,
    CLASS_PERFORMANCE_WEIGHTS,
    DISEASE_CO_OCCURRENCE_ENABLED,
    DISEASE_CO_OCCURRENCE_MATRIX,
    LEARNABLE_FUSION_ENABLED,
    LEARNABLE_FUSION_MLP_HIDDEN,
    MC_DROPOUT_PASSES,
    MODEL_LIST,
    MODEL_WEIGHTS,
    STACKING_ENABLED,
    STACKING_META_LEARNER,
    TEMPERATURE_SCALING,
    TTA_AGGREGATION_METHOD,
    UNCERTAINTY_AWARE_TTA_ENABLED,
    UNCERTAINTY_THRESHOLD,
)
from retina_app.services.exceptions import InferenceError
from retina_app.services.model_manager import DEVICE
from retina_app.services.transforms import TRANSFORM, TRANSFORMS

logger = logging.getLogger("retina_app")


def _apply_disease_co_occurrence(probs):
    """Apply disease co-occurrence prior to prediction probabilities.

    Uses Bayesian updating with co-occurrence matrix to adjust predictions
    based on medical knowledge of disease relationships.

    Reference: Disease co-occurrence patterns in retinal imaging studies.
    """
    if not DISEASE_CO_OCCURRENCE_ENABLED:
        return probs

    co_matrix = DISEASE_CO_OCCURRENCE_MATRIX
    if not co_matrix:
        return probs

    probs_arr = np.array(probs)
    n_classes = len(probs_arr)

    # Build co-occurrence prior from matrix
    prior = np.zeros(n_classes)
    for i, cat in enumerate(CATEGORIES):
        if cat in co_matrix:
            prior[i] = co_matrix[cat][i]

    # Normalize prior
    prior_sum = prior.sum()
    if prior_sum > 0:
        prior = prior / prior_sum

    # Bayesian update: posterior ∝ likelihood × prior
    # Use log-space for numerical stability
    log_likelihood = np.log(probs_arr + 1e-10)
    log_prior = np.log(prior + 1e-10)
    log_posterior = log_likelihood + log_prior

    # Convert back to probabilities
    log_posterior = log_posterior - np.max(log_posterior)
    posterior = np.exp(log_posterior)
    posterior = posterior / posterior.sum()

    return posterior.tolist()


def _compute_tta_uncertainty(all_probs):
    """Compute uncertainty across TTA variants using variance.

    Lower variance = higher confidence in prediction consistency.
    """
    if len(all_probs) < 2:
        return 0.0

    probs_arr = np.stack(all_probs)
    variance = np.var(probs_arr, axis=0)
    return float(np.mean(variance))


class LearnableFusion(nn.Module):
    """Learnable fusion module for ensemble predictions.

    Instead of fixed weights, this module learns optimal combination
    of model predictions based on confidence and agreement.

    Reference: Res101-MViT-Ens (2026) - End-to-end dynamic learnable weight fusion
    """

    def __init__(self, n_models=5, n_classes=4, hidden_dim=64, dropout=0.3):
        super().__init__()

        # Input: probs from all models + confidences
        input_dim = n_models * n_classes + n_models

        self.fusion_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_models),
            nn.Softmax(dim=-1),
        )

        self.n_models = n_models
        self.n_classes = n_classes

    def forward(self, model_probs_list, model_confidences):
        """Compute learnable fusion weights.

        Args:
            model_probs_list: List of [n_classes] probability tensors
            model_confidences: [n_models] confidence scores

        Returns:
            [n_models] fusion weights
        """
        all_probs = torch.cat(model_probs_list, dim=-1)
        combined = torch.cat([all_probs, model_confidences], dim=-1)
        weights = self.fusion_net(combined)
        return weights


# Singleton learnable fusion
_learnable_fusion = None


def get_learnable_fusion():
    """Get or create learnable fusion singleton."""
    global _learnable_fusion
    if _learnable_fusion is None:
        from retina_app.constants import CATEGORIES

        _learnable_fusion = LearnableFusion(
            n_models=len(MODEL_LIST),
            n_classes=len(CATEGORIES),
            hidden_dim=LEARNABLE_FUSION_MLP_HIDDEN,
        )
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "learnable_fusion.pth"
        )
        if os.path.exists(model_path):
            try:
                _learnable_fusion.load_state_dict(torch.load(model_path, map_location=DEVICE))
                _learnable_fusion.to(DEVICE)
                logger.info("Loaded learnable fusion from disk")
            except Exception as e:
                logger.warning(f"Failed to load learnable fusion: {e}")
    return _learnable_fusion


# ── Stacking Meta-Learner ─────────────────────────────────────────────────────
class StackingMetaLearner:
    """Learns optimal combination of model predictions via stacking.

    Collects base model predictions on a calibration set, then trains a
    meta-learner (logistic regression or small MLP) to combine them.
    """

    def __init__(self, meta_learner_type="logistic", n_classes=4):
        self.meta_learner_type = meta_learner_type
        self.n_classes = n_classes
        self.meta_model = None
        self.model_names = None
        self.scaler = None
        self._is_trained = False

    def _build_meta_model(self, n_features):
        """Build meta-learner model."""
        if self.meta_learner_type == "mlp":
            return nn.Sequential(
                nn.Linear(n_features, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, self.n_classes),
            )
        else:
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(max_iter=1000, multi_class="multinomial", C=1.0, solver="lbfgs")

    def train(self, model_predictions, true_labels, model_names):
        """Train meta-learner on calibration set.

        Args:
            model_predictions: dict {model_name: (n_samples, n_classes) probs}
            true_labels: (n_samples,) ground truth
            model_names: list of model names
        """
        self.model_names = model_names

        # Stack predictions: each model's probs become features
        # Input: [p_m1_c1, p_m1_c2, ..., p_m1_cK, p_m2_c1, ..., p_mN_cK]
        stacked = np.concatenate([model_predictions[name] for name in model_names], axis=1)
        n_features = stacked.shape[1]

        if self.meta_learner_type == "logistic":
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
            stacked = self.scaler.fit_transform(stacked)
            self.meta_model = self._build_meta_model(n_features)
            self.meta_model.fit(stacked, true_labels)
        else:
            # MLP training with PyTorch
            self.meta_model = self._build_meta_model(n_features)
            X = torch.tensor(stacked, dtype=torch.float32)
            y = torch.tensor(true_labels, dtype=torch.long)

            optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            self.meta_model.train()
            for epoch in range(200):
                optimizer.zero_grad()
                outputs = self.meta_model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

        self._is_trained = True
        logger.info(f"Stacking meta-learner trained: {self.meta_learner_type}, {n_features} features")

    def predict(self, model_probs_dict):
        """Predict using trained meta-learner.

        Args:
            model_probs_dict: dict {model_name: (n_classes,) probs}

        Returns:
            (n_classes,) averaged probabilities
        """
        if not self._is_trained or self.meta_model is None:
            return None

        stacked = np.concatenate([model_probs_dict[name] for name in self.model_names], axis=0).reshape(1, -1)

        if self.meta_learner_type == "logistic" and self.scaler is not None:
            stacked = self.scaler.transform(stacked)
            proba = self.meta_model.predict_proba(stacked)[0]
            return proba
        else:
            X = torch.tensor(stacked, dtype=torch.float32)
            self.meta_model.eval()
            with torch.no_grad():
                logits = self.meta_model(X)
                probs = F.softmax(logits, dim=1).numpy()[0]
            return probs

    def save(self, path):
        """Save trained meta-learner."""
        data = {
            "meta_learner_type": self.meta_learner_type,
            "n_classes": self.n_classes,
            "model_names": self.model_names,
            "is_trained": self._is_trained,
        }
        if self.meta_learner_type == "logistic":
            data["meta_model"] = self.meta_model
            data["scaler"] = self.scaler
        else:
            data["meta_model_state"] = self.meta_model.state_dict() if self.meta_model else None

        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Stacking meta-learner saved to {path}")

    @classmethod
    def load(cls, path):
        """Load trained meta-learner."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        learner = cls(
            meta_learner_type=data["meta_learner_type"],
            n_classes=data["n_classes"],
        )
        learner.model_names = data["model_names"]
        learner._is_trained = data["is_trained"]

        if data["meta_learner_type"] == "logistic":
            learner.meta_model = data["meta_model"]
            learner.scaler = data.get("scaler")
        else:
            n_features = len(data["model_names"]) * data["n_classes"]
            learner.meta_model = learner._build_meta_model(n_features)
            if data.get("meta_model_state"):
                learner.meta_model.load_state_dict(data["meta_model_state"])

        return learner


# Singleton meta-learner
_meta_learner = None


def get_meta_learner():
    """Get or create the stacking meta-learner singleton."""
    global _meta_learner
    if _meta_learner is None:
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "stacking_meta_learner.pkl"
        )
        if os.path.exists(model_path):
            try:
                _meta_learner = StackingMetaLearner.load(model_path)
                logger.info("Loaded stacking meta-learner from disk")
            except Exception as e:
                logger.warning(f"Failed to load meta-learner: {e}")
                _meta_learner = StackingMetaLearner(
                    meta_learner_type=STACKING_META_LEARNER,
                    n_classes=len(CATEGORIES),
                )
        else:
            _meta_learner = StackingMetaLearner(
                meta_learner_type=STACKING_META_LEARNER,
                n_classes=len(CATEGORIES),
            )
    return _meta_learner


def apply_temperature_scaling(logits, temperature=TEMPERATURE_SCALING):
    """Apply temperature scaling for confidence calibration."""
    if temperature == 1.0:
        return torch.softmax(logits, dim=1)
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=1)


def _predict_single_model(model, image_path, use_tta=False):
    """Run inference on a single model with optional TTA."""
    with Image.open(image_path) as pil_img:
        image = pil_img.convert("RGB")
        if use_tta:
            all_logits = []
            all_probs = []
            failed_transforms = []

            for transform_name, transform in TRANSFORMS.items():
                try:
                    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        output = model(input_tensor)
                        if isinstance(output, tuple):
                            output = output[0]
                        if len(output.shape) == 4:
                            output = output.view(output.size(0), output.size(1), -1).squeeze(-1).squeeze(0)
                        elif len(output.shape) == 3:
                            output = output.view(output.size(0), output.size(1), -1).squeeze(-1).squeeze(0)
                        elif len(output.shape) == 2:
                            output = output.squeeze(0)
                        if output.numel() != len(CATEGORIES):
                            if output.numel() % len(CATEGORIES) == 0:
                                spatial_size = output.numel() // len(CATEGORIES)
                                h = w = int(spatial_size**0.5)
                                if h * w == spatial_size:
                                    output = output.view(len(CATEGORIES), h, w).mean(dim=(1, 2))
                                else:
                                    continue
                            else:
                                continue
                        all_logits.append(output.detach().cpu().numpy())
                        probs = F.softmax(output, dim=0)
                        all_probs.append(probs.cpu().numpy())
                except Exception:
                    failed_transforms.append(transform_name)
                    continue

            if not all_probs:
                raise InferenceError(f"All TTA transforms failed for {image_path}")

            # Compute TTA uncertainty
            tta_uncertainty = _compute_tta_uncertainty(all_probs)

            # Uncertainty-aware aggregation (BayTTA-inspired)
            if UNCERTAINTY_AWARE_TTA_ENABLED and tta_uncertainty > UNCERTAINTY_THRESHOLD:
                # High variance → weight predictions inversely to their variance
                logger.debug(f"High TTA variance: {tta_uncertainty:.4f}, using uncertainty-aware aggregation")

                # Weight by inverse variance (lower variance = higher weight)
                weights = [1.0 / (np.var(p) + 1e-6) for p in all_probs]
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]

                avg_probs = np.zeros(len(CATEGORIES))
                for i, probs in enumerate(all_probs):
                    avg_probs += np.array(probs) * weights[i]
                avg_probs = avg_probs / avg_probs.sum()
            elif TTA_AGGREGATION_METHOD == "geometric":
                stacked = np.stack(all_probs)
                log_probs = np.log(stacked + 1e-10)
                avg_log = np.mean(log_probs, axis=0)
                avg_probs = np.exp(avg_log)
                avg_probs = avg_probs / np.sum(avg_probs)
            else:
                avg_probs = sum(all_probs) / len(all_probs)

            # Apply disease co-occurrence matrix
            avg_probs = _apply_disease_co_occurrence(avg_probs.tolist())

            avg_logits = np.mean(np.stack(all_logits), axis=0).tolist()
            max_idx = max(range(len(avg_probs)), key=lambda i: avg_probs[i])
            confidence = avg_probs[max_idx]

            result = {
                "label": CATEGORIES[max_idx],
                "confidence": float(confidence),
                "probabilities": avg_probs,
                "logits": avg_logits,
                "tta_uncertainty": tta_uncertainty,
            }
            if failed_transforms:
                result["warnings"] = [f"Transform {t} failed" for t in failed_transforms]
            return result
        else:
            input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(input_tensor)
                if isinstance(output, tuple):
                    output = output[0]
                if len(output.shape) == 4:
                    output = output.view(output.size(0), output.size(1), -1).squeeze(-1).squeeze(0)
                elif len(output.shape) == 3:
                    output = output.view(output.size(0), output.size(1), -1).squeeze(-1).squeeze(0)
                elif len(output.shape) == 2:
                    output = output.squeeze(0)
                if output.numel() != len(CATEGORIES):
                    if output.numel() % len(CATEGORIES) == 0:
                        spatial_size = output.numel() // len(CATEGORIES)
                        h = w = int(spatial_size**0.5)
                        if h * w == spatial_size:
                            output = output.view(len(CATEGORIES), h, w).mean(dim=(1, 2))
                        else:
                            raise InferenceError(f"Cannot reshape output {output.numel()} to {len(CATEGORIES)} classes")
                    else:
                        raise InferenceError(f"Unexpected output size: {output.numel()}")

                logits = output.detach().cpu().numpy().tolist()
                probabilities = F.softmax(output, dim=0)
                confidence, predicted_idx = torch.max(probabilities, dim=0)

            # Apply disease co-occurrence matrix to single prediction too
            probs_list = probabilities.tolist()
            probs_list = _apply_disease_co_occurrence(probs_list)
            max_idx = probs_list.index(max(probs_list))

            return {
                "label": CATEGORIES[max_idx],
                "confidence": float(max(probs_list)),
                "probabilities": probs_list,
                "logits": logits,
            }


def predict_models_parallel(models, image_path, use_tta, executor):
    """Run parallel inference on multiple models."""
    predictions = []

    def predict_wrapper(model_type_and_model):
        model_type, model = model_type_and_model
        try:
            pred = _predict_single_model(model, image_path, use_tta=use_tta)
            return (model_type, pred, None)
        except Exception as exc:
            return (model_type, None, exc)

    results = list(executor.map(predict_wrapper, list(models.items())))
    for model_type, pred, error in results:
        if pred is not None:
            predictions.append((model_type, pred))
        else:
            logger.warning(f"Model {model_type} prediction failed: {error}")

    if not predictions:
        raise InferenceError("All models failed to make predictions")
    return predictions


def ensemble_predictions(predictions):
    """Combine predictions using stacking (if available) or weighted averaging."""
    if not predictions:
        raise ValueError("No predictions to ensemble")

    n_models = len(predictions)
    n_classes = len(CATEGORIES)

    # Try stacking meta-learner first
    if STACKING_ENABLED:
        meta_learner = get_meta_learner()
        if meta_learner._is_trained:
            try:
                model_probs = {}
                for model_type, pred in predictions:
                    model_probs[model_type] = np.array(pred["probabilities"])

                stacked_probs = meta_learner.predict(model_probs)
                if stacked_probs is not None:
                    # Apply disease co-occurrence to stacking output
                    stacked_probs = _apply_disease_co_occurrence(stacked_probs.tolist())

                    max_idx = int(np.argmax(stacked_probs))
                    avg_confidence = sum(pred["confidence"] for _, pred in predictions) / n_models
                    ensemble_uncertainty = 1.0 - max(stacked_probs)

                    return {
                        "label": CATEGORIES[max_idx],
                        "confidence": float(stacked_probs[max_idx]),
                        "avg_model_confidence": avg_confidence,
                        "n_models": n_models,
                        "probabilities": stacked_probs,
                        "uncertainty": ensemble_uncertainty,
                        "method": "stacking",
                    }
            except Exception as exc:
                logger.warning(f"Stacking meta-learner failed, falling back to weighted averaging: {exc}")

    # Try learnable fusion (Res101-MViT-Ens inspired)
    if LEARNABLE_FUSION_ENABLED:
        try:
            fusion = get_learnable_fusion()
            if fusion is not None and fusion.training:
                model_probs_list = []
                model_confidences = []

                for model_type, pred in predictions:
                    probs = torch.tensor(pred["probabilities"], dtype=torch.float32).to(DEVICE)
                    conf = torch.tensor([pred["confidence"]], dtype=torch.float32).to(DEVICE)
                    model_probs_list.append(probs)
                    model_confidences.append(conf)

                # Pad to expected n_models if needed
                while len(model_probs_list) < len(MODEL_LIST):
                    model_probs_list.append(torch.zeros(n_classes, dtype=torch.float32).to(DEVICE))
                    model_confidences.append(torch.tensor([0.0], dtype=torch.float32).to(DEVICE))

                model_confidences = torch.cat(model_confidences[: len(MODEL_LIST)])

                weights = fusion(model_probs_list, model_confidences)

                # Apply learned weights
                weighted_probs = np.zeros(n_classes)
                for i, (model_type, pred) in enumerate(predictions):
                    if i < len(MODEL_LIST):
                        probs = np.array(pred["probabilities"])
                        weighted_probs += probs * weights[i].item()

                weighted_probs = _apply_disease_co_occurrence(weighted_probs.tolist())
                max_idx = int(np.argmax(weighted_probs))
                avg_confidence = sum(pred["confidence"] for _, pred in predictions) / n_models

                return {
                    "label": CATEGORIES[max_idx],
                    "confidence": float(weighted_probs[max_idx]),
                    "avg_model_confidence": avg_confidence,
                    "n_models": n_models,
                    "probabilities": weighted_probs,
                    "uncertainty": 1.0 - max(weighted_probs),
                    "method": "learnable_fusion",
                }
        except Exception as exc:
            logger.debug(f"Learnable fusion not available, falling back to weighted averaging: {exc}")

    # Fallback: per-class dynamic weighted averaging
    weighted_probs = [0.0] * n_classes
    class_weights = [0.0] * n_classes
    model_details = []

    raw_weights = []
    valid_predictions = []
    for model_type, pred in predictions:
        probs = pred["probabilities"]
        confidence = pred["confidence"]
        predicted_class = pred["label"]

        base_weight = MODEL_WEIGHTS.get(model_type, 1.0 / n_models)
        if predicted_class in CLASS_PERFORMANCE_WEIGHTS:
            class_weight = CLASS_PERFORMANCE_WEIGHTS[predicted_class].get(model_type, base_weight)
        else:
            class_weight = base_weight

        confidence_boost = 1.0 + (confidence - 0.5) * 0.2
        final_weight = class_weight * confidence_boost
        raw_weights.append(final_weight)
        valid_predictions.append((model_type, pred, final_weight, predicted_class, confidence))

    total_weight = sum(raw_weights)
    if total_weight > 0:
        raw_weights = [w / total_weight for w in raw_weights]

    for i, (model_type, pred, _, predicted_class, confidence) in enumerate(valid_predictions):
        probs = pred["probabilities"]
        final_weight = raw_weights[i]
        for j, p in enumerate(probs):
            weighted_probs[j] += p * final_weight
            class_weights[j] += final_weight
        model_details.append(
            {
                "model": model_type,
                "label": predicted_class,
                "confidence": confidence,
                "weight": final_weight,
            }
        )

    normalized_probs = [weighted_probs[i] / class_weights[i] if class_weights[i] > 0 else 0 for i in range(n_classes)]
    total = sum(normalized_probs)
    if total > 0:
        normalized_probs = [p / total for p in normalized_probs]

    # Apply disease co-occurrence matrix
    normalized_probs = _apply_disease_co_occurrence(normalized_probs)

    max_idx = int(np.argmax(normalized_probs))
    avg_confidence = sum(pred["confidence"] for _, pred in predictions) / n_models
    ensemble_uncertainty = 1.0 - max(normalized_probs)

    return {
        "label": CATEGORIES[max_idx],
        "confidence": normalized_probs[max_idx],
        "avg_model_confidence": avg_confidence,
        "n_models": n_models,
        "probabilities": normalized_probs,
        "uncertainty": ensemble_uncertainty,
        "method": "weighted_average",
    }


def predict_with_uncertainty_ensemble(models, image_path, model_weights=None, n_passes=MC_DROPOUT_PASSES):
    """Run MC Dropout uncertainty quantification across an ensemble."""
    from retina_app.services.uncertainty import mc_dropout_ensemble

    return mc_dropout_ensemble(models, image_path, model_weights=model_weights, n_passes=n_passes)


def detect_model_disagreement(predictions):
    """Detect when models disagree on the predicted class."""
    if len(predictions) < 2:
        label = predictions[0][1]["label"] if predictions else None
        return {
            "disagreement": False,
            "agreement_level": 1.0,
            "dominant_class": label,
            "class_votes": {label: 1} if label else {},
            "disagreeing_models": [],
            "model_predictions": {},
        }

    class_votes = {}
    model_predictions = {}
    for model_type, pred in predictions:
        label = pred["label"]
        class_votes[label] = class_votes.get(label, 0) + 1
        model_predictions[model_type] = label

    n_models = len(predictions)
    dominant_class = max(class_votes, key=class_votes.get)
    dominant_count = class_votes[dominant_class]
    agreement_level = dominant_count / n_models
    disagreeing_models = [mt for mt, label in model_predictions.items() if label != dominant_class]

    return {
        "disagreement": len(disagreeing_models) > 0,
        "agreement_level": round(agreement_level, 3),
        "dominant_class": dominant_class,
        "class_votes": class_votes,
        "disagreeing_models": disagreeing_models,
        "model_predictions": model_predictions,
    }


def selective_ensemble(predictions, min_agreement=0.5):
    """Selective ensemble that filters out outlier predictions."""
    if len(predictions) <= 2:
        result = ensemble_predictions(predictions)
        result["selective_ensemble"] = False
        result["agreement_level"] = 1.0
        return result

    analysis = detect_model_disagreement(predictions)

    if not analysis["disagreement"]:
        return ensemble_predictions(predictions)

    if analysis["agreement_level"] >= min_agreement:
        majority_class = analysis["dominant_class"]
        filtered = [(mt, pred) for mt, pred in predictions if pred["label"] == majority_class]
        if len(filtered) >= 2:
            result = ensemble_predictions(filtered)
            result["selective_ensemble"] = True
            result["original_n_models"] = len(predictions)
            result["filtered_n_models"] = len(filtered)
            result["agreement_level"] = analysis["agreement_level"]
            return result

    result = ensemble_predictions(predictions)
    result["selective_ensemble"] = False
    result["agreement_level"] = analysis["agreement_level"]
    return result
