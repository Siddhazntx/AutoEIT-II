import logging
import math
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class HeuristicScorer:
    """
    Interpretable baseline scorer for AutoEIT.

    Combines extracted features using a weighted linear equation and maps the
    continuous score to the final 0-4 rubric using configurable thresholds.
    """

    def __init__(self, config: dict):
        scoring_config = config.get("scoring_engine", {})

        self.weights = scoring_config.get(
            "weights",
            {
                "nli_margin": 0.50,
                "sbert_similarity": 0.30,
                "lemma_recall": 0.20,
            },
        )

        self.thresholds = scoring_config.get(
            "heuristic_thresholds",
            [0.20, 0.40, 0.60, 0.85],
        )

        self._validate_weights()
        self._normalize_weights()
        self._validate_thresholds()

        logger.info(f"Heuristic scorer initialized with weights: {self.weights}")
        logger.info(f"Heuristic scorer thresholds: {self.thresholds}")

    def _validate_weights(self) -> None:
        """
        Validate that the configured weights are usable.
        """
        if not isinstance(self.weights, dict) or not self.weights:
            raise ValueError("Heuristic weights must be a non-empty dictionary.")

        for feature_name, weight in self.weights.items():
            if not isinstance(feature_name, str) or not feature_name.strip():
                raise ValueError("Each weight key must be a non-empty string.")

            if not isinstance(weight, (int, float)):
                raise ValueError(f"Weight for feature '{feature_name}' must be numeric.")

            if weight < 0:
                raise ValueError(f"Weight for feature '{feature_name}' cannot be negative.")

        if sum(self.weights.values()) <= 0:
            raise ValueError("Sum of heuristic weights must be greater than 0.")

    def _normalize_weights(self) -> None:
        """
        Normalize weights so they sum to 1.0.
        """
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def _validate_thresholds(self) -> None:
        """
        Validate threshold structure and values.
        Expected format:
            [T0_to_1, T1_to_2, T2_to_3, T3_to_4]
        """
        if not isinstance(self.thresholds, list) or len(self.thresholds) != 4:
            raise ValueError(
                "heuristic_thresholds must be a list of 4 increasing numeric values."
            )

        for t in self.thresholds:
            if not isinstance(t, (int, float)):
                raise ValueError("All heuristic thresholds must be numeric.")
            if not (0.0 <= t <= 1.0):
                raise ValueError("All heuristic thresholds must be within [0.0, 1.0].")

        if self.thresholds != sorted(self.thresholds):
            raise ValueError("heuristic_thresholds must be strictly increasing / sorted.")

    def _safe_feature_value(self, value: Any) -> float:
        """
        Safely convert a feature value to float.
        Invalid values fall back to 0.0.
        """
        if value is None:
            return 0.0

        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0

        if math.isnan(value) or math.isinf(value):
            return 0.0

        return value

    def _clamp_score(self, score: float) -> float:
        """
        Clamp final raw score to [0.0, 1.0] for rubric stability.
        """
        return max(0.0, min(1.0, score))

    def compute_feature_contributions(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Return the weighted contribution of each configured feature.
        """
        contributions: Dict[str, float] = {}

        for feature_name, weight in self.weights.items():
            feature_val = self._safe_feature_value(features.get(feature_name, 0.0))
            contributions[feature_name] = round(feature_val * weight, 4)

        return contributions

    def compute_raw_score(self, features: Dict[str, Any]) -> float:
        """
        Compute the weighted continuous score from extracted features.
        """
        raw_score = 0.0

        for feature_name, weight in self.weights.items():
            feature_val = self._safe_feature_value(features.get(feature_name, 0.0))
            raw_score += feature_val * weight

        raw_score = self._clamp_score(raw_score)
        return round(raw_score, 4)

    def map_to_rubric(self, raw_score: float) -> int:
        """
        Map continuous score to integer rubric score in [0, 4].
        """
        t1, t2, t3, t4 = self.thresholds

        if raw_score < t1:
            return 0
        if raw_score < t2:
            return 1
        if raw_score < t3:
            return 2
        if raw_score < t4:
            return 3
        return 4

    def score_single(
        self,
        features: Dict[str, Any],
        return_details: bool = False,
        early_gate_score: Optional[int] = None,
    ) -> Any:
        """
        Score a single item.

        If early_gate_score is 0 or 4, that score is returned directly.
        Otherwise, feature-based heuristic scoring is applied.
        """
        if early_gate_score in {0, 4}:
            if return_details:
                return {
                    "predicted_score": early_gate_score,
                    "raw_continuous_score": None,
                    "reason": "early_gate_override",
                    "applied_weights": self.weights,
                    "feature_contributions": {},
                    "input_features": features,
                }
            return early_gate_score

        raw_score = self.compute_raw_score(features)
        final_score = self.map_to_rubric(raw_score)
        contributions = self.compute_feature_contributions(features)

        if return_details:
            return {
                "predicted_score": final_score,
                "raw_continuous_score": raw_score,
                "applied_weights": self.weights,
                "feature_contributions": contributions,
                "input_features": features,
            }

        return final_score

    def score_batch(
        self,
        batch_features: List[Dict[str, Any]],
        early_gate_scores: Optional[List[Optional[int]]] = None,
    ) -> List[int]:
        """
        Score a batch of feature dictionaries.

        If early_gate_scores is provided, matching rows with score 0 or 4
        are overridden directly.
        """
        if not batch_features:
            return []

        if early_gate_scores is not None and len(early_gate_scores) != len(batch_features):
            raise ValueError(
                "Length of early_gate_scores must match length of batch_features."
            )

        logger.info(f"Applying heuristic scoring to {len(batch_features)} items...")

        final_scores: List[int] = []

        for idx, features in enumerate(batch_features):
            gate_score = early_gate_scores[idx] if early_gate_scores is not None else None
            final_scores.append(self.score_single(features, return_details=False, early_gate_score=gate_score))

        return final_scores