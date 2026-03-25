import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import yaml
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.data.data_loader import EITDataLoader
from src.preprocessing.preprocessor import EITPreprocessor
from src.features.feature_extractor import MasterFeatureExtractor
from src.scoring.heuristic_scorer import HeuristicScorer
from src.scoring.thresholding import QWKThresholdOptimizer
from src.scoring.ordinal_model import OrdinalScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class AutoEITPipeline:
    """
    Master controller for the AutoEIT scoring system.

    Pipeline stages:
    1. Load configuration and datasets
    2. Preprocess text + apply early quality gate
    3. Extract linguistic / semantic features only where needed
    4. Train:
        A) Heuristic scorer + threshold optimizer
        B) Ordinal ML scorer
    5. Evaluate both approaches on a labeled validation split
    6. Run inference on unlabeled holdout data
    7. Save experiment artifacts
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        logger.info("Initializing AutoEIT Master Pipeline...")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.data_loader = EITDataLoader(self.config)
        self.preprocessor = EITPreprocessor(self.config)
        self.feature_extractor = MasterFeatureExtractor(self.config)

        self.heuristic_scorer = HeuristicScorer(self.config)
        self.optimizer = QWKThresholdOptimizer(self.config)
        self.ordinal_model = OrdinalScorer(self.config)

        paths_cfg = self.config.get("paths", {})
        self.output_dir = Path(paths_cfg.get("processed_data_dir", "data/processed/"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        eval_cfg = self.config.get("evaluation", {})
        self.validation_size = float(eval_cfg.get("validation_size", 0.2))
        self.random_state = int(self.config.get("project", {}).get("seed", 42))
        self.stratify_validation = bool(eval_cfg.get("stratify_validation", True))

    def _compute_metrics(self, true_labels: List[int], preds: List[int]) -> Dict[str, float]:
        return {
            "qwk": round(float(cohen_kappa_score(true_labels, preds, weights="quadratic")), 4),
            "accuracy": round(float(accuracy_score(true_labels, preds)), 4),
            "macro_f1": round(float(f1_score(true_labels, preds, average="macro", zero_division=0)), 4),
        }

    def _detect_score_column(self, df: pd.DataFrame, dataset_name: str) -> Optional[str]:
        logger.info(f"Raw columns found in {dataset_name}: {df.columns.tolist()}")

        exact_candidates = {
            "score",
            "human score",
            "human_score",
            "ortega score",
            "human grade",
            "grade",
            "final score",
            "rater score",
        }

        for col in df.columns:
            col_clean = str(col).strip().lower()
            if col_clean in exact_candidates:
                logger.info(f"✅ Successfully mapped true labels to column: '{col}'")
                return col

        for col in df.columns:
            col_clean = str(col).strip().lower()
            if "score" in col_clean or "grade" in col_clean:
                logger.info(f"✅ Fallback-mapped true labels to column: '{col}'")
                return col

        logger.warning(f"⚠️ No score column found in {dataset_name}. Rows will be marked as unlabeled.")
        return None

    def _process_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str,
    ) -> Tuple[List[Dict[str, Any]], List[Optional[int]], List[Optional[int]], pd.DataFrame]:
        if df.empty:
            logger.warning(f"{dataset_name} DataFrame is empty.")
            return [], [], [], pd.DataFrame()

        logger.info(f"Processing {dataset_name} ({len(df)} rows)...")
        score_col = self._detect_score_column(df, dataset_name)

        if score_col is not None:
            logger.info(f"{dataset_name} sample raw score values: {df[score_col].head(10).tolist()}")

        row_records: List[Dict[str, Any]] = []
        pairs_for_models: List[Tuple[str, str]] = []
        feature_row_indices: List[int] = []

        gate_0_count = 0
        gate_4_count = 0
        gate_pass_count = 0

        for row_idx, (_, row) in enumerate(df.iterrows()):
            stimulus = row["stimulus"]
            transcription = row["transcription"]

            clean_stim, clean_resp, gate = self.preprocessor.preprocess_pair(
                stimulus,
                transcription,
            )

            human_score: Optional[int] = None
            if score_col is not None:
                try:
                    raw_score = row[score_col]
                    if pd.notna(raw_score):
                        score_str = str(raw_score).strip()
                        if score_str != "":
                            score_num = pd.to_numeric(score_str, errors="coerce")
                            if pd.notna(score_num):
                                human_score = int(float(score_num))
                except (KeyError, TypeError, ValueError):
                    human_score = None

            early_gate_score: Optional[int] = gate if gate in {0, 4} else None

            if early_gate_score == 0:
                gate_0_count += 1
            elif early_gate_score == 4:
                gate_4_count += 1
            else:
                gate_pass_count += 1
                pairs_for_models.append((clean_stim, clean_resp))
                feature_row_indices.append(row_idx)

            row_records.append(
                {
                    "row_index": row_idx,
                    "participant_id": row.get("participant_id", "N/A"),
                    "sheet_name": row.get("sheet_name", "N/A"),
                    "stimulus": stimulus,
                    "transcription": transcription,
                    "clean_stimulus": clean_stim,
                    "clean_response": clean_resp,
                    "human_score": human_score,
                    "early_gate_score": early_gate_score,
                }
            )

        valid_label_count = sum(1 for r in row_records if r["human_score"] is not None)
        logger.info(f"{dataset_name} labeled rows detected: {valid_label_count}/{len(row_records)}")

        logger.info(
            f"{dataset_name} gate summary | "
            f"score_0={gate_0_count}, score_4={gate_4_count}, feature_extraction_needed={gate_pass_count}"
        )

        features_batch: List[Dict[str, Any]] = [{} for _ in range(len(row_records))]

        if pairs_for_models:
            logger.info(f"Extracting AI features for {len(pairs_for_models)} unresolved {dataset_name} rows...")
            extracted_features = self.feature_extractor.extract_features_batch(pairs_for_models)

            for original_idx, feature_dict in zip(feature_row_indices, extracted_features):
                features_batch[original_idx] = feature_dict

        audit_rows = []
        for record, feat in zip(row_records, features_batch):
            merged = dict(record)
            merged.update(feat)
            audit_rows.append(merged)

        audit_df = pd.DataFrame(audit_rows)

        true_labels = [r["human_score"] for r in row_records]
        early_gate_scores = [r["early_gate_score"] for r in row_records]

        return features_batch, true_labels, early_gate_scores, audit_df

    def _save_results(self, df: pd.DataFrame, filename: str) -> None:
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved artifact: {output_path}")

    def _split_labeled_tuning_data(
        self,
        feats: List[Dict[str, Any]],
        labels: List[Optional[int]],
        gates: List[Optional[int]],
    ):
        labeled_idx = [i for i, y in enumerate(labels) if y is not None]

        if not labeled_idx:
            raise ValueError("No labeled rows found in tuning set.")

        labeled_feats = [feats[i] for i in labeled_idx]
        labeled_labels = [int(labels[i]) for i in labeled_idx]
        labeled_gates = [gates[i] for i in labeled_idx]
        labeled_original_idx = labeled_idx.copy()

        stratify_labels = labeled_labels if self.stratify_validation else None

        try:
            split = train_test_split(
                labeled_feats,
                labeled_labels,
                labeled_gates,
                labeled_original_idx,
                test_size=self.validation_size,
                random_state=self.random_state,
                stratify=stratify_labels,
            )
        except ValueError:
            logger.warning("Stratified split failed. Falling back to non-stratified split.")
            split = train_test_split(
                labeled_feats,
                labeled_labels,
                labeled_gates,
                labeled_original_idx,
                test_size=self.validation_size,
                random_state=self.random_state,
                stratify=None,
            )

        (
            train_feats,
            val_feats,
            train_labels,
            val_labels,
            train_gates,
            val_gates,
            train_idx,
            val_idx,
        ) = split

        return train_feats, val_feats, train_labels, val_labels, train_gates, val_gates, train_idx, val_idx

    def run_experiment(self) -> None:
        print("\n" + "=" * 70)
        print("🚀 STARTING AUTOEIT GSoC PIPELINE")
        print("=" * 70)

        tuning_df, holdout_df = self.data_loader.load_data()
        if tuning_df.empty:
            logger.error("Failed to load tuning dataset. Halting pipeline.")
            return

        tune_feats, tune_labels, tune_gates, tune_audit = self._process_dataset(tuning_df, "Tuning Set")

        if holdout_df.empty:
            logger.warning("Holdout dataset is empty. Continuing with validation-only experiment.")
            holdout_feats, holdout_labels, holdout_gates, holdout_audit = [], [], [], pd.DataFrame()
        else:
            holdout_feats, holdout_labels, holdout_gates, holdout_audit = self._process_dataset(holdout_df, "Holdout Set")

        try:
            (
                train_feats,
                val_feats,
                train_labels,
                val_labels,
                train_gates,
                val_gates,
                train_idx,
                val_idx,
            ) = self._split_labeled_tuning_data(tune_feats, tune_labels, tune_gates)
        except ValueError as exc:
            logger.error(str(exc))
            return

        logger.info(
            f"Labeled tuning split complete | train={len(train_labels)} rows, validation={len(val_labels)} rows"
        )

        print("\n" + "-" * 70)
        print("🧪 PHASE A: TRAINING ON LABELED TUNING SPLIT")
        print("-" * 70)

        logger.info("Training Approach A: Heuristic + QWK threshold optimization")

        optimization_feats = []
        optimization_labels = []

        for feat, label, gate in zip(train_feats, train_labels, train_gates):
            if gate is None:
                optimization_feats.append(feat)
                optimization_labels.append(label)

        logger.info(
            f"Heuristic threshold optimization rows: {len(optimization_labels)} "
            f"(excluded gate-resolved rows: {len(train_labels) - len(optimization_labels)})"
        )

        if not optimization_feats:
            logger.warning("No unresolved training rows found for heuristic threshold optimization. Using current thresholds.")
            optimal_thresholds = self.heuristic_scorer.thresholds
        else:
            train_raw_scores = [self.heuristic_scorer.compute_raw_score(f) for f in optimization_feats]

            initial_metrics = self.optimizer.evaluate(
                raw_scores=train_raw_scores,
                true_labels=optimization_labels,
                thresholds=self.optimizer.initial_guess,
            )
            logger.info(f"Initial heuristic optimization metrics: {initial_metrics}")

            optimal_thresholds = self.optimizer.fit(train_raw_scores, optimization_labels)

        self.heuristic_scorer.thresholds = optimal_thresholds

        heuristic_train_preds = self.heuristic_scorer.score_batch(
            train_feats,
            early_gate_scores=train_gates,
        )
        optimized_metrics = self._compute_metrics(train_labels, heuristic_train_preds)
        logger.info(f"Optimized heuristic training metrics: {optimized_metrics}")

        logger.info("Training Approach B: Ordinal classifier")
        self.ordinal_model.fit(train_feats, train_labels)

        print("\n" + "-" * 70)
        print("📊 PHASE B: VALIDATION EVALUATION")
        print("-" * 70)

        heuristic_val_preds = self.heuristic_scorer.score_batch(
            val_feats,
            early_gate_scores=val_gates,
        )
        heuristic_metrics = self._compute_metrics(val_labels, heuristic_val_preds)

        ordinal_val_preds = self.ordinal_model.score_batch(
            val_feats,
            early_gate_scores=val_gates,
        )
        ordinal_metrics = self._compute_metrics(val_labels, ordinal_val_preds)

        logger.info(f"Heuristic validation metrics: {heuristic_metrics}")
        logger.info(f"Ordinal validation metrics: {ordinal_metrics}")

        val_results_df = pd.DataFrame(
            {
                "original_row_index": val_idx,
                "human_score": val_labels,
                "early_gate_score": val_gates,
                "heuristic_pred": heuristic_val_preds,
                "ordinal_pred": ordinal_val_preds,
                "heuristic_raw_score": [self.heuristic_scorer.compute_raw_score(f) for f in val_feats],
            }
        )
        self._save_results(val_results_df, "validation_predictions.csv")

        holdout_summary = {
            "rows": 0,
            "labeled_rows": 0,
        }

        if holdout_feats:
            logger.info("Running inference on holdout set...")
            holdout_heuristic_preds = self.heuristic_scorer.score_batch(
                holdout_feats,
                early_gate_scores=holdout_gates,
            )
            holdout_ordinal_preds = self.ordinal_model.score_batch(
                holdout_feats,
                early_gate_scores=holdout_gates,
            )

            holdout_audit = holdout_audit.copy()
            holdout_audit["heuristic_pred"] = holdout_heuristic_preds
            holdout_audit["ordinal_pred"] = holdout_ordinal_preds
            holdout_audit["heuristic_raw_score"] = [
                self.heuristic_scorer.compute_raw_score(f) for f in holdout_feats
            ]

            self._save_results(holdout_audit, "holdout_predictions.csv")

            holdout_summary["rows"] = len(holdout_audit)
            holdout_summary["labeled_rows"] = sum(1 for y in holdout_labels if y is not None)

        self._save_results(tune_audit, "tuning_audit.csv")

        summary_df = pd.DataFrame(
            [
                {
                    "evaluation_split": "validation_from_tuning",
                    "approach": "heuristic",
                    "qwk": heuristic_metrics["qwk"],
                    "accuracy": heuristic_metrics["accuracy"],
                    "macro_f1": heuristic_metrics["macro_f1"],
                },
                {
                    "evaluation_split": "validation_from_tuning",
                    "approach": "ordinal_ml",
                    "qwk": ordinal_metrics["qwk"],
                    "accuracy": ordinal_metrics["accuracy"],
                    "macro_f1": ordinal_metrics["macro_f1"],
                },
            ]
        )
        self._save_results(summary_df, "experiment_summary.csv")

        print("\n" + "=" * 70)
        print("🏆 FINAL A/B TEST RESULTS")
        print("=" * 70)
        print(f"{'Metric':<20} | {'Approach A (Heuristic)':<24} | {'Approach B (Ordinal ML)':<24}")
        print("-" * 76)
        print(f"{'QWK (Validation)':<20} | {heuristic_metrics['qwk']:<24.4f} | {ordinal_metrics['qwk']:<24.4f}")
        print(f"{'Accuracy':<20} | {heuristic_metrics['accuracy']:<24.4f} | {ordinal_metrics['accuracy']:<24.4f}")
        print(f"{'Macro F1':<20} | {heuristic_metrics['macro_f1']:<24.4f} | {ordinal_metrics['macro_f1']:<24.4f}")
        print("=" * 76)

        winner = (
            "Approach A (Heuristic)"
            if heuristic_metrics["qwk"] >= ordinal_metrics["qwk"]
            else "Approach B (Ordinal ML)"
        )
        print(f"\n💡 Conclusion: {winner} achieved the higher QWK on the validation split.")
        print(f"📁 Saved outputs to: {self.output_dir}")

        if holdout_summary["rows"] > 0:
            print(
                f"📝 Holdout inference completed for {holdout_summary['rows']} rows "
                f"(labeled rows available: {holdout_summary['labeled_rows']})."
            )
        print()

        logger.info(f"Best heuristic thresholds: {optimal_thresholds}")
        logger.info(f"Heuristic validation metrics: {heuristic_metrics}")
        logger.info(f"Ordinal validation metrics: {ordinal_metrics}")


if __name__ == "__main__":
    pipeline = AutoEITPipeline()
    pipeline.run_experiment()