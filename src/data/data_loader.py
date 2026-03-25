import logging
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class EITDataLoader:
    """
    AutoEIT data loader for Excel-based datasets.

    Responsibilities:
    - Read tuning and holdout .xlsx workbooks directly
    - Parse all relevant sheets
    - Standardize schema across inconsistent tabs
    - Clean and validate rows
    - Cache flattened outputs as CSV for faster reloads
    """

    def __init__(self, config: dict):
        self.config = config

        paths_cfg = config.get("paths", {})
        dataset_cfg = config.get("dataset", {})

        self.raw_dir = Path(paths_cfg.get("raw_data_dir", "data/raw/"))
        self.cache_dir = Path(paths_cfg.get("cache_dir", "data/cache/"))

        self.tuning_file = self.raw_dir / dataset_cfg.get(
            "tuning_set",
            "Example_EIT Transcription and Scoring Sheet.xlsx",
        )
        self.holdout_file = self.raw_dir / dataset_cfg.get(
            "holdout_test_set",
            "AutoEIT Sample Transcriptions for Scoring.xlsx",
        )

        self.force_reload = bool(dataset_cfg.get("force_reload", False))
        self.allowed_score_range = tuple(dataset_cfg.get("allowed_score_range", [0, 4]))

        self.ignored_tabs = [
            "info",
            "instruction",
            "instructions",
            "stimuli",
            "key",
            "rubric",
            "scores",
            "participant information",
            "metadata",
        ]

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _is_cache_valid(self, source_file: Path, cache_file: Path) -> bool:
        """
        Cache is valid only if:
        - cache exists
        - source exists
        - force_reload is False
        - cache is newer than or same age as source
        """
        if self.force_reload:
            return False

        if not source_file.exists() or not cache_file.exists():
            return False

        try:
            return cache_file.stat().st_mtime >= source_file.stat().st_mtime
        except OSError:
            return False

    def _extract_participant_id(self, sheet_name: str) -> str:
        """
        Extract participant ID from sheet name.
        Example:
            'Participant 1' -> 'P1'
            '29_vA' -> 'P29'
            fallback -> sanitized sheet name
        """
        match = re.search(r"\d+", sheet_name)
        if match:
            return f"P{match.group(0)}"

        sanitized = re.sub(r"\s+", "_", sheet_name.strip())
        return sanitized or "unknown_participant"

    def _clean_stimulus(self, text: str) -> str:
        """
        Removes trailing syllable/item counts like:
            'Quiero cortarme el pelo (7)' -> 'Quiero cortarme el pelo'
        """
        if pd.isna(text):
            return ""
        cleaned = re.sub(r"\s*\(\d+\)\s*$", "", str(text))
        return cleaned.strip()

    def _normalize_string_series(self, series: pd.Series) -> pd.Series:
        return series.astype(str).str.strip()

    def _should_ignore_sheet(self, sheet_name: str) -> bool:
        lower_name = sheet_name.lower().strip()
        return any(token in lower_name for token in self.ignored_tabs)

    # ---------------------------------------------------------
    # Column Standardization
    # ---------------------------------------------------------
    def _standardize_columns(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """
        Map messy Excel column names to internal stable names:
        - stimulus
        - transcription
        - human_score
        """
        renamed = df.copy()
        col_map = {}

        for col in renamed.columns:
            if not isinstance(col, str):
                continue

            c = col.lower().strip()

            # Stronger priority matching first
            if c in {"stimulus", "target", "prompt", "sentence", "source sentence"}:
                col_map[col] = "stimulus"
            elif "stimulus" in c or "target" in c or "prompt" in c:
                col_map[col] = "stimulus"

            elif c in {"transcription", "response", "student response", "student transcription"}:
                col_map[col] = "transcription"
            elif "transcription" in c or "response" in c or "student" in c:
                if "transcription" not in col_map.values():
                    col_map[col] = "transcription"

            elif c in {"score", "human score", "rater score", "rubric score", "grade"}:
                if "human_score" not in col_map.values():
                    col_map[col] = "human_score"
            elif "score" in c or "rubric" in c or "grade" in c:
                if "human_score" not in col_map.values():
                    col_map[col] = "human_score"

        renamed = renamed.rename(columns=col_map)
        renamed["participant_id"] = self._extract_participant_id(sheet_name)
        renamed["sheet_name"] = sheet_name

        logger.debug(f"Sheet '{sheet_name}' column mapping: {col_map}")
        return renamed

    # ---------------------------------------------------------
    # Row Validation / Cleaning
    # ---------------------------------------------------------
    def _validate_and_clean_sheet(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        is_tuning: bool,
    ) -> pd.DataFrame:
        """
        Validate schema, clean rows, and return standardized subset.
        """
        required = {"stimulus", "transcription"}
        if not required.issubset(df.columns):
            logger.warning(
                f"Skipping sheet '{sheet_name}': missing required columns. "
                f"Found columns: {list(df.columns)}"
            )
            return pd.DataFrame()

        cols_to_keep: List[str] = ["participant_id", "sheet_name", "stimulus", "transcription"]
        if "human_score" in df.columns:
            cols_to_keep.append("human_score")

        out = df[cols_to_keep].copy()

        # Drop true NaNs before string conversion
        before_dropna = len(out)
        out = out.dropna(subset=["stimulus", "transcription"])
        dropped_na = before_dropna - len(out)

        # Normalize strings
        out["stimulus"] = self._normalize_string_series(out["stimulus"])
        out["transcription"] = self._normalize_string_series(out["transcription"])

        # Remove blank strings
        before_blank_filter = len(out)
        out = out[
            (out["stimulus"] != "") &
            (out["transcription"] != "")
        ].copy()
        dropped_blank = before_blank_filter - len(out)

        # Clean stimulus only
        out["stimulus"] = out["stimulus"].apply(self._clean_stimulus)

        # Handle score if present
        invalid_score_count = 0
        out_of_range_count = 0

        if "human_score" in out.columns:
            out["human_score"] = pd.to_numeric(out["human_score"], errors="coerce")
            invalid_score_count = int(out["human_score"].isna().sum())

            if is_tuning:
                min_score, max_score = self.allowed_score_range
                out_of_range_mask = (
                    out["human_score"].notna() &
                    ((out["human_score"] < min_score) | (out["human_score"] > max_score))
                )
                out_of_range_count = int(out_of_range_mask.sum())

                if out_of_range_count > 0:
                    logger.warning(
                        f"Sheet '{sheet_name}' has {out_of_range_count} scores outside "
                        f"expected rubric range [{min_score}, {max_score}]. These rows will be dropped."
                    )
                    out = out.loc[~out_of_range_mask].copy()

        logger.info(
            f"Sheet '{sheet_name}': kept {len(out)} rows "
            f"(dropped NaN: {dropped_na}, dropped blank: {dropped_blank}, "
            f"invalid scores: {invalid_score_count}, out-of-range scores: {out_of_range_count})"
        )

        return out

    # ---------------------------------------------------------
    # Workbook Processing
    # ---------------------------------------------------------
    def _process_excel_file(self, file_path: Path, is_tuning: bool = False) -> pd.DataFrame:
        """
        Read one workbook, process all valid sheets, and return a flattened DataFrame.
        """
        file_stem = file_path.stem
        cache_path = self.cache_dir / f"{file_stem}_cached.csv"

        if self._is_cache_valid(file_path, cache_path):
            logger.info(f"Loading cached standardized data from: {cache_path}")
            return pd.read_csv(cache_path)

        if not file_path.exists():
            logger.error(f"Workbook not found: {file_path}")
            return pd.DataFrame()

        logger.info(f"Processing workbook: {file_path.name}")

        try:
            excel_data = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")
        except Exception as exc:
            logger.error(f"Failed to read workbook '{file_path.name}': {exc}")
            return pd.DataFrame()

        total_sheets = len(excel_data)
        skipped_sheets = 0
        valid_sheets = 0
        processed_dfs: List[pd.DataFrame] = []

        for sheet_name, df in excel_data.items():
            if self._should_ignore_sheet(sheet_name):
                skipped_sheets += 1
                logger.debug(f"Skipping metadata sheet: {sheet_name}")
                continue

            standardized = self._standardize_columns(df, sheet_name)
            cleaned = self._validate_and_clean_sheet(
                standardized,
                sheet_name=sheet_name,
                is_tuning=is_tuning,
            )

            if not cleaned.empty:
                processed_dfs.append(cleaned)
                valid_sheets += 1

        if not processed_dfs:
            logger.error(f"No valid rows found in workbook: {file_path.name}")
            return pd.DataFrame()

        final_df = pd.concat(processed_dfs, ignore_index=True)

        before_dedup = len(final_df)
        final_df = final_df.drop_duplicates(
            subset=["participant_id", "sheet_name", "stimulus", "transcription"]
        ).reset_index(drop=True)
        deduped = before_dedup - len(final_df)

        # Tuning set must have valid human_score
        if is_tuning:
            if "human_score" not in final_df.columns:
                raise ValueError(
                    f"Tuning workbook '{file_path.name}' does not contain a usable human_score column."
                )

            before_score_drop = len(final_df)
            final_df = final_df.dropna(subset=["human_score"]).copy()
            dropped_missing_scores = before_score_drop - len(final_df)

            # Optional: cast to int only if all scores are integer-valued
            if not final_df.empty:
                try:
                    if (final_df["human_score"] % 1 == 0).all():
                        final_df["human_score"] = final_df["human_score"].astype(int)
                except Exception:
                    pass

            logger.info(f"Dropped {dropped_missing_scores} tuning rows missing human_score.")
        else:
            dropped_missing_scores = 0

        expected_cols = {"participant_id", "sheet_name", "stimulus", "transcription"}
        if is_tuning:
            expected_cols.add("human_score")

        missing = expected_cols - set(final_df.columns)
        if missing:
            raise ValueError(f"Final dataset missing required columns: {missing}")

        logger.info(
            f"Workbook summary for '{file_path.name}': "
            f"total sheets={total_sheets}, skipped={skipped_sheets}, "
            f"valid={valid_sheets}, final rows={len(final_df)}, deduplicated={deduped}, "
            f"dropped missing scores={dropped_missing_scores}"
        )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(cache_path, index=False)
        logger.info(f"Saved cached standardized data to: {cache_path}")

        return final_df

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both tuning and holdout datasets.
        """
        logger.info("=== Starting Data Ingestion Phase ===")

        tuning_df = self._process_excel_file(self.tuning_file, is_tuning=True)
        holdout_df = self._process_excel_file(self.holdout_file, is_tuning=False)

        logger.info(
            f"=== Data Ingestion Complete | tuning_rows={len(tuning_df)}, "
            f"holdout_rows={len(holdout_df)} ==="
        )

        return tuning_df, holdout_df