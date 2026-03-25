import logging
from typing import List, Dict, Optional, Any

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score

logger = logging.getLogger(__name__)


class QWKThresholdOptimizer:
    """
    Optimizes ordinal decision thresholds for a 0-4 rubric by maximizing
    Quadratic Weighted Kappa (QWK) between raw continuous scores and human labels.

    Designed for AutoEIT heuristic/ordinal score mapping.
    """

    def __init__(self, config: Optional[dict] = None):
        config = config or {}

        optimization_cfg = config.get("optimization", {})
        grid_cfg = optimization_cfg.get("grid_search", {})
        thresholds_cfg = config.get("thresholds", {})
        evaluation_cfg = config.get("evaluation", {})

        self.num_classes = 5
        self.num_thresholds = self.num_classes - 1

        # Initial guess from config thresholds if available
        self.initial_guess = [
            float(thresholds_cfg.get("T1", 0.20)),
            float(thresholds_cfg.get("T2", 0.40)),
            float(thresholds_cfg.get("T3", 0.60)),
            float(thresholds_cfg.get("T4", 0.85)),
        ]

        # Optional bounds
        bounds_cfg = grid_cfg.get("bounds", {})
        self.bounds = [
            (
                float(bounds_cfg.get("T1_min", 0.00)),
                float(bounds_cfg.get("T1_max", 0.40)),
            ),
            (
                float(bounds_cfg.get("T2_min", 0.20)),
                float(bounds_cfg.get("T2_max", 0.70)),
            ),
            (
                float(bounds_cfg.get("T3_min", 0.40)),
                float(bounds_cfg.get("T3_max", 0.90)),
            ),
            (
                float(bounds_cfg.get("T4_min", 0.60)),
                float(bounds_cfg.get("T4_max", 1.00)),
            ),
        ]

        self.maxiter = int(optimization_cfg.get("maxiter", 1000))
        self.primary_metric = evaluation_cfg.get("primary_metric", "quadratic_weighted_kappa")

        self.best_thresholds: List[float] = []
        self.best_qwk = -1.0
        self.optimization_result: Optional[Any] = None

        self._validate_threshold_list(self.initial_guess, name="initial_guess")
        self._validate_bounds()

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------
    def _validate_threshold_list(self, thresholds: List[float], name: str = "thresholds") -> None:
        if not isinstance(thresholds, list) or len(thresholds) != self.num_thresholds:
            raise ValueError(
                f"{name} must be a list of {self.num_thresholds} numeric thresholds."
            )

        for t in thresholds:
            if not isinstance(t, (int, float)):
                raise ValueError(f"All values in {name} must be numeric.")
            if not (0.0 <= float(t) <= 1.0):
                raise ValueError(f"All values in {name} must lie within [0.0, 1.0].")

        if any(thresholds[i] >= thresholds[i + 1] for i in range(len(thresholds) - 1)):
            raise ValueError(f"{name} must be strictly increasing.")

    def _validate_bounds(self) -> None:
        if len(self.bounds) != self.num_thresholds:
            raise ValueError(f"Bounds must contain {self.num_thresholds} entries.")

        for idx, pair in enumerate(self.bounds):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"Bound at index {idx} must be a (min, max) pair.")

            low, high = float(pair[0]), float(pair[1])

            if low < 0.0 or high > 1.0 or low >= high:
                raise ValueError(
                    f"Invalid bound at index {idx}: ({low}, {high}). "
                    "Bounds must satisfy 0.0 <= low < high <= 1.0."
                )

    def _validate_fit_inputs(self, raw_scores: List[float], true_labels: List[int]) -> None:
        if raw_scores is None or true_labels is None:
            raise ValueError("raw_scores and true_labels must not be None.")

        if len(raw_scores) == 0 or len(true_labels) == 0:
            raise ValueError("raw_scores and true_labels must not be empty.")

        if len(raw_scores) != len(true_labels):
            raise ValueError("raw_scores and true_labels must have the same length.")

        invalid_labels = [y for y in true_labels if y not in {0, 1, 2, 3, 4}]
        if invalid_labels:
            raise ValueError("true_labels must contain only integer rubric values in {0,1,2,3,4}.")

    # ---------------------------------------------------------
    # Core logic
    # ---------------------------------------------------------
    def _apply_thresholds(self, raw_scores: np.ndarray, thresholds: List[float]) -> np.ndarray:
        """
        Map continuous scores to ordinal classes using numpy.digitize.
        Example:
            thresholds = [0.2, 0.4, 0.6, 0.85]
            raw_score = 0.5 -> class 2
        """
        thresholds = np.array(thresholds, dtype=float)
        raw_scores = np.asarray(raw_scores, dtype=float)

        raw_scores = np.clip(raw_scores, 0.0, 1.0)
        preds = np.digitize(raw_scores, thresholds, right=False)

        # np.digitize already yields values in [0, len(thresholds)]
        return preds.astype(int)

    def _qwk_objective_function(
        self,
        thresholds: List[float],
        raw_scores: np.ndarray,
        true_labels: np.ndarray,
    ) -> float:
        """
        Objective for SciPy minimization: minimize negative QWK.
        Penalizes invalid threshold orderings heavily.
        """
        thresholds = list(map(float, thresholds))

        if any(thresholds[i] >= thresholds[i + 1] for i in range(len(thresholds) - 1)):
            return 999.0

        if any(t < 0.0 or t > 1.0 for t in thresholds):
            return 999.0

        preds = self._apply_thresholds(raw_scores, thresholds)
        qwk = cohen_kappa_score(true_labels, preds, weights="quadratic")

        if np.isnan(qwk):
            return 999.0

        return -float(qwk)

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def fit(self, raw_scores: List[float], true_labels: List[int]) -> List[float]:
        """
        Optimize thresholds to maximize QWK.
        """
        self._validate_fit_inputs(raw_scores, true_labels)

        raw_scores_np = np.asarray(raw_scores, dtype=float)
        true_labels_np = np.asarray(true_labels, dtype=int)

        logger.info(f"Starting QWK threshold optimization on {len(raw_scores_np)} samples...")
        logger.info(f"Initial thresholds: {self.initial_guess}")
        logger.info(f"Bounds: {self.bounds}")

        # Baseline before optimization
        baseline_qwk = -self._qwk_objective_function(self.initial_guess, raw_scores_np, true_labels_np)
        logger.info(f"Baseline QWK before optimization: {baseline_qwk:.4f}")

        result = minimize(
            fun=self._qwk_objective_function,
            x0=np.array(self.initial_guess, dtype=float),
            args=(raw_scores_np, true_labels_np),
            method="Powell",
            bounds=self.bounds,
            options={"maxiter": self.maxiter, "disp": False},
        )

        self.optimization_result = result

        if result.success:
            candidate_thresholds = sorted(np.round(result.x, 4).tolist())

            # Safety: if optimizer returns weird ordering after sorting, validate again
            try:
                self._validate_threshold_list(candidate_thresholds, name="optimized_thresholds")
                self.best_thresholds = candidate_thresholds
                self.best_qwk = -float(result.fun)
                logger.info(f"Optimization successful. Best QWK: {self.best_qwk:.4f}")
                logger.info(f"Optimized thresholds: {self.best_thresholds}")
            except ValueError as exc:
                logger.warning(
                    f"Optimization produced invalid thresholds ({candidate_thresholds}). "
                    f"Falling back to initial guess. Reason: {exc}"
                )
                self.best_thresholds = self.initial_guess
                self.best_qwk = baseline_qwk
        else:
            logger.warning("Optimization failed to converge. Falling back to initial guess.")
            self.best_thresholds = self.initial_guess
            self.best_qwk = baseline_qwk

        return self.best_thresholds

    def predict(self, raw_scores: List[float], thresholds: Optional[List[float]] = None) -> List[int]:
        """
        Convert raw continuous scores into discrete rubric predictions.
        """
        eval_thresholds = thresholds or self.best_thresholds or self.initial_guess
        self._validate_threshold_list(eval_thresholds, name="prediction_thresholds")

        raw_np = np.asarray(raw_scores, dtype=float)
        preds = self._apply_thresholds(raw_np, eval_thresholds)
        return preds.tolist()

    def evaluate(
        self,
        raw_scores: List[float],
        true_labels: List[int],
        thresholds: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate predictions under a given threshold set.
        """
        self._validate_fit_inputs(raw_scores, true_labels)

        eval_thresholds = thresholds or self.best_thresholds or self.initial_guess
        self._validate_threshold_list(eval_thresholds, name="evaluation_thresholds")

        raw_np = np.asarray(raw_scores, dtype=float)
        true_np = np.asarray(true_labels, dtype=int)

        preds = self._apply_thresholds(raw_np, eval_thresholds)

        qwk = cohen_kappa_score(true_np, preds, weights="quadratic")
        acc = accuracy_score(true_np, preds)
        macro_f1 = f1_score(true_np, preds, average="macro", zero_division=0)

        return {
            "qwk": round(float(qwk), 4),
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
        }

    def compare_initial_vs_optimized(
        self,
        raw_scores: List[float],
        true_labels: List[int],
    ) -> Dict[str, Dict[str, float]]:
        """
        Convenience method to compare baseline thresholds vs optimized thresholds.
        """
        initial_metrics = self.evaluate(raw_scores, true_labels, thresholds=self.initial_guess)
        optimized_thresholds = self.best_thresholds or self.initial_guess
        optimized_metrics = self.evaluate(raw_scores, true_labels, thresholds=optimized_thresholds)

        return {
            "initial": initial_metrics,
            "optimized": optimized_metrics,
        }