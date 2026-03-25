import logging
import math
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score

logger = logging.getLogger(__name__)


class FrankAndHallOrdinalWrapper(BaseEstimator, ClassifierMixin):
    """
    Frank & Hall (2001) ordinal reduction:
    convert an ordinal K-class problem into K-1 binary problems:
        P(y > c_0), P(y > c_1), ..., P(y > c_{K-2})
    """

    def __init__(self, base_estimator: BaseEstimator):
        self.base_estimator = base_estimator
        self.estimators_: List[BaseEstimator] = []
        self.classes_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._validate_fit_inputs(X, y)

        self.classes_ = np.sort(np.unique(y))
        self.estimators_ = []

        for i in range(len(self.classes_) - 1):
            binary_y = (y > self.classes_[i]).astype(int)
            estimator = clone(self.base_estimator)
            estimator.fit(X, binary_y)
            self.estimators_.append(estimator)

        return self

    def _validate_fit_inputs(self, X: np.ndarray, y: np.ndarray) -> None:
        if X is None or y is None:
            raise ValueError("X and y must not be None.")

        if len(X) == 0 or len(y) == 0:
            raise ValueError("X and y must not be empty.")

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")

        unique_classes = np.sort(np.unique(y))
        if len(unique_classes) <= 1:
            raise ValueError("Ordinal classification requires at least 2 distinct classes.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Recover class probabilities from cumulative binary probabilities.

        If classes are [0,1,2,3,4], we estimate:
            P(y=0) = 1 - P(y>0)
            P(y=1) = P(y>0) - P(y>1)
            ...
            P(y=4) = P(y>3)
        """
        if len(self.estimators_) == 0:
            raise RuntimeError("Model must be fitted before calling predict_proba().")

        X = np.asarray(X, dtype=float)

        prob_greater_than = np.array(
            [est.predict_proba(X)[:, 1] for est in self.estimators_]
        ).T

        probs = np.zeros((X.shape[0], len(self.classes_)), dtype=float)

        probs[:, 0] = 1.0 - prob_greater_than[:, 0]

        for i in range(1, len(self.classes_) - 1):
            probs[:, i] = prob_greater_than[:, i - 1] - prob_greater_than[:, i]

        probs[:, -1] = prob_greater_than[:, -1]

        probs = np.clip(probs, 0.0, 1.0)

        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        probs = probs / row_sums

        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(self.estimators_) == 0:
            raise RuntimeError("Model must be fitted before calling predict().")

        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


class OrdinalScorer:
    """
    AutoEIT-facing ordinal ML scorer.

    Responsibilities:
    - vectorize feature dictionaries in stable column order
    - train ordinal model
    - predict rubric scores
    - merge early-gate overrides
    - evaluate with QWK / Accuracy / Macro-F1
    """

    def __init__(self, config: dict):
        scoring_config = config.get("scoring_engine", {})
        project_config = config.get("project", {})

        self.feature_names = scoring_config.get(
            "ml_features",
            [
                "nli_margin",
                "nli_entailment",
                "nli_contradiction",
                "sbert_similarity",
                "lemma_recall",
            ],
        )

        self.model_type = scoring_config.get("ml_model_type", "logistic_regression")
        self.random_state = int(project_config.get("seed", 42))

        self.model = self._build_model(scoring_config)
        self.is_fitted = False

    def _build_model(self, scoring_config: dict) -> FrankAndHallOrdinalWrapper:
        """
        Build backend model and wrap it in the ordinal reduction architecture.
        """
        if self.model_type == "random_forest":
            base_model = RandomForestClassifier(
                n_estimators=int(scoring_config.get("rf_n_estimators", 100)),
                max_depth=scoring_config.get("rf_max_depth", 5),
                min_samples_split=int(scoring_config.get("rf_min_samples_split", 2)),
                min_samples_leaf=int(scoring_config.get("rf_min_samples_leaf", 1)),
                class_weight="balanced",
                random_state=self.random_state,
            )
            logger.info("Initializing Ordinal Model with Random Forest backend.")
        else:
            base_model = LogisticRegression(
                class_weight="balanced",
                random_state=self.random_state,
                max_iter=int(scoring_config.get("lr_max_iter", 1000)),
                C=float(scoring_config.get("lr_C", 1.0)),
                solver=scoring_config.get("lr_solver", "lbfgs"),
            )
            logger.info("Initializing Ordinal Model with Logistic Regression backend.")

        return FrankAndHallOrdinalWrapper(base_estimator=base_model)

    def _safe_float(self, value: Any) -> float:
        if value is None:
            return 0.0

        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0

        if math.isnan(value) or math.isinf(value):
            return 0.0

        return value

    def _vectorize_features(self, batch_features: List[Dict[str, Any]]) -> np.ndarray:
        """
        Convert list of feature dictionaries into 2D NumPy matrix using fixed feature order.
        """
        X = np.zeros((len(batch_features), len(self.feature_names)), dtype=float)

        for i, features in enumerate(batch_features):
            for j, fname in enumerate(self.feature_names):
                X[i, j] = self._safe_float(features.get(fname, 0.0))

        return X

    def _validate_training_inputs(
        self,
        batch_features: List[Dict[str, Any]],
        true_labels: List[int],
    ) -> None:
        if not batch_features:
            raise ValueError("batch_features must not be empty.")

        if len(batch_features) != len(true_labels):
            raise ValueError("batch_features and true_labels must have the same length.")

        invalid = [y for y in true_labels if y not in {0, 1, 2, 3, 4}]
        if invalid:
            raise ValueError("true_labels must contain only rubric scores in {0,1,2,3,4}.")

        unique_classes = sorted(set(true_labels))
        if len(unique_classes) < 2:
            raise ValueError("Training requires at least 2 distinct label classes.")

    def fit(self, batch_features: List[Dict[str, Any]], true_labels: List[int]) -> None:
        """
        Train the ordinal ML model.
        """
        self._validate_training_inputs(batch_features, true_labels)

        logger.info(f"Training Ordinal Scorer on {len(batch_features)} samples...")
        X = self._vectorize_features(batch_features)
        y = np.asarray(true_labels, dtype=int)

        self.model.fit(X, y)
        self.is_fitted = True

        logger.info("Ordinal Scorer training complete.")

    def predict_proba_batch(self, batch_features: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """
        Return class probabilities for each item.
        """
        if not self.is_fitted:
            raise RuntimeError("OrdinalScorer must be trained before prediction.")

        if not batch_features:
            return []

        X = self._vectorize_features(batch_features)
        probs = self.model.predict_proba(X)

        results: List[Dict[str, float]] = []
        for row in probs:
            results.append(
                {
                    "prob_0": round(float(row[0]), 4),
                    "prob_1": round(float(row[1]), 4),
                    "prob_2": round(float(row[2]), 4),
                    "prob_3": round(float(row[3]), 4),
                    "prob_4": round(float(row[4]), 4),
                }
            )
        return results

    def score_single(
        self,
        features: Dict[str, Any],
        early_gate_score: Optional[int] = None,
        return_details: bool = False,
    ) -> Any:
        """
        Score a single feature dictionary with optional early-gate override.
        """
        if early_gate_score in {0, 4}:
            if return_details:
                return {
                    "predicted_score": int(early_gate_score),
                    "reason": "early_gate_override",
                    "probabilities": None,
                    "feature_names": self.feature_names,
                    "input_features": features,
                }
            return int(early_gate_score)

        if not self.is_fitted:
            raise RuntimeError("OrdinalScorer must be trained before prediction.")

        X = self._vectorize_features([features])
        pred = int(self.model.predict(X)[0])
        probs = self.model.predict_proba(X)[0]

        if return_details:
            return {
                "predicted_score": pred,
                "probabilities": {
                    "prob_0": round(float(probs[0]), 4),
                    "prob_1": round(float(probs[1]), 4),
                    "prob_2": round(float(probs[2]), 4),
                    "prob_3": round(float(probs[3]), 4),
                    "prob_4": round(float(probs[4]), 4),
                },
                "feature_names": self.feature_names,
                "input_features": features,
            }

        return pred

    def score_batch(
        self,
        batch_features: List[Dict[str, Any]],
        early_gate_scores: Optional[List[Optional[int]]] = None,
    ) -> List[int]:
        """
        Score a batch, merging early-gate overrides where applicable.
        """
        if not self.is_fitted:
            raise RuntimeError("OrdinalScorer must be trained by calling fit() before scoring.")

        if not batch_features:
            return []

        if early_gate_scores is not None and len(early_gate_scores) != len(batch_features):
            raise ValueError("Length of early_gate_scores must match batch_features.")

        X = self._vectorize_features(batch_features)
        ml_predictions = self.model.predict(X).tolist()

        final_scores: List[int] = []
        for i in range(len(batch_features)):
            gate_score = early_gate_scores[i] if early_gate_scores is not None else None
            if gate_score in {0, 4}:
                final_scores.append(int(gate_score))
            else:
                final_scores.append(int(ml_predictions[i]))

        return final_scores

    def evaluate(
        self,
        batch_features: List[Dict[str, Any]],
        true_labels: List[int],
        early_gate_scores: Optional[List[Optional[int]]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate ordinal scorer predictions.
        """
        preds = self.score_batch(batch_features, early_gate_scores)

        true_np = np.asarray(true_labels, dtype=int)
        pred_np = np.asarray(preds, dtype=int)

        return {
            "qwk": round(float(cohen_kappa_score(true_np, pred_np, weights="quadratic")), 4),
            "accuracy": round(float(accuracy_score(true_np, pred_np)), 4),
            "macro_f1": round(float(f1_score(true_np, pred_np, average="macro", zero_division=0)), 4),
        }

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Return a lightweight model summary for debugging / experiment tracking.
        """
        summary: Dict[str, Any] = {
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
        }

        if not self.is_fitted:
            return summary

        if self.model_type == "logistic_regression":
            coef_summary = []
            for idx, estimator in enumerate(self.model.estimators_):
                if hasattr(estimator, "coef_"):
                    coef_summary.append(
                        {
                            "binary_task": f"y > {idx}",
                            "coefficients": {
                                fname: round(float(coef), 4)
                                for fname, coef in zip(self.feature_names, estimator.coef_[0])
                            },
                            "intercept": round(float(estimator.intercept_[0]), 4),
                        }
                    )
            summary["binary_models"] = coef_summary

        elif self.model_type == "random_forest":
            importance_summary = []
            for idx, estimator in enumerate(self.model.estimators_):
                if hasattr(estimator, "feature_importances_"):
                    importance_summary.append(
                        {
                            "binary_task": f"y > {idx}",
                            "feature_importances": {
                                fname: round(float(imp), 4)
                                for fname, imp in zip(self.feature_names, estimator.feature_importances_)
                            },
                        }
                    )
            summary["binary_models"] = importance_summary

        return summary