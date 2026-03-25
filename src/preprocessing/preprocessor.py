import logging
import re
import unicodedata
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class EITPreprocessor:
    def __init__(self, config: dict):
        """
        Initializes the preprocessor using rules from config.yaml.
        Expected input: full config dict or preprocessing subsection.
        """
        self.config = config.get("preprocessing", config)

        # Noise tags like [pause], [cough], [laugh]
        noise_tags = [re.escape(tag) for tag in self.config.get("noise_tags_to_remove", [])]
        self.tags_pattern: Optional[re.Pattern] = (
            re.compile(r"|".join(noise_tags), re.IGNORECASE) if noise_tags else None
        )

        # Gibberish markers like xxx, xx, x
        gibberish_markers = [
            rf"\b{re.escape(marker)}\b"
            for marker in self.config.get("gibberish_markers", [])
        ]
        self.gibberish_pattern: Optional[re.Pattern] = (
            re.compile(r"|".join(gibberish_markers), re.IGNORECASE) if gibberish_markers else None
        )

        # Broken restart like: dis- disminuido
        self.restart_pattern = re.compile(r"\b\w+-\s+(?=\w+\b)", re.IGNORECASE)

        # Single-letter stutter like: m- muy
        self.single_letter_stutter_pattern = re.compile(r"\b\w-\s+(?=\w+\b)", re.IGNORECASE)

        # Extra whitespace cleanup
        self.multispace_pattern = re.compile(r"\s+")

        # Punctuation removal while preserving word chars/spaces
        self.punctuation_pattern = re.compile(r"[^\w\s]", re.UNICODE)

    def _remove_accents(self, text: str) -> str:
        """
        Normalize accents: 'está' -> 'esta'
        """
        return "".join(
            ch for ch in unicodedata.normalize("NFD", text)
            if unicodedata.category(ch) != "Mn"
        )

    def _normalize_whitespace(self, text: str) -> str:
        return self.multispace_pattern.sub(" ", text).strip()

    def _coerce_text(self, text) -> str:
        """
        Safely convert to string while treating missing values as empty.
        """
        if pd.isna(text):
            return ""
        return str(text)

    def _is_effectively_empty(self, text: str) -> bool:
        """
        Detect responses that are empty or nearly empty after cleaning.
        """
        if not text:
            return True

        alnum_only = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
        return len(alnum_only) == 0

    def clean_text(self, text) -> str:
        """
        Cleans text using configurable preprocessing rules.
        """
        cleaned = self._coerce_text(text)
        if not cleaned:
            return ""

        if self.config.get("lowercase", True):
            cleaned = cleaned.lower()

        if self.tags_pattern:
            cleaned = self.tags_pattern.sub(" ", cleaned)

        if self.gibberish_pattern:
            cleaned = self.gibberish_pattern.sub(" ", cleaned)

        # Remove restart fragments like: dis- disminuido
        cleaned = self.restart_pattern.sub("", cleaned)

        # Remove short stutters like: m- muy
        cleaned = self.single_letter_stutter_pattern.sub("", cleaned)

        if self.config.get("remove_punctuation", True):
            cleaned = self.punctuation_pattern.sub(" ", cleaned)

        if self.config.get("normalize_accents", True):
            cleaned = self._remove_accents(cleaned)

        cleaned = self._normalize_whitespace(cleaned)

        return "" if self._is_effectively_empty(cleaned) else cleaned

    def early_quality_gate(self, clean_stimulus: str, clean_response: str) -> int:
        """
        Early scoring shortcut.

        Returns:
            4  -> exact semantic-form match after cleaning
            0  -> empty / pure gibberish after cleaning
           -1  -> continue to downstream scoring
        """
        stimulus = self._normalize_whitespace(self._coerce_text(clean_stimulus))
        response = self._normalize_whitespace(self._coerce_text(clean_response))

        if self._is_effectively_empty(response):
            return 0

        if stimulus and stimulus == response:
            return 4

        return -1

    def preprocess_pair(self, stimulus, response) -> tuple[str, str, int]:
        """
        Convenience helper for full preprocessing flow.
        Returns cleaned stimulus, cleaned response, and gate decision.
        """
        clean_stimulus = self.clean_text(stimulus)
        clean_response = self.clean_text(response)
        gate_score = self.early_quality_gate(clean_stimulus, clean_response)
        return clean_stimulus, clean_response, gate_score