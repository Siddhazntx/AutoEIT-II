import logging
from typing import Dict, Set

import spacy

logger = logging.getLogger(__name__)


class LinguisticFeatureExtractor:
    def __init__(self, config: dict):
        """
        spaCy-based linguistic feature extractor for AutoEIT.
        Computes lemma and idea-unit overlap signals between stimulus and response.
        """
        models_config = config.get("models", {})
        features_config = config.get("features", {})

        self.model_name = models_config.get("spacy_model", "es_core_news_lg")
        self.content_pos = set(
            features_config.get(
                "content_pos_tags",
                ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]
            )
        )
        self.remove_stopwords_for_ideas = features_config.get(
            "remove_stopwords_for_ideas", True
        )

        self.nlp = self._load_spacy()

    def _load_spacy(self) -> spacy.Language:
        """
        Load the configured spaCy model.
        """
        try:
            logger.info(f"Loading spaCy model: {self.model_name}")
            return spacy.load(self.model_name)
        except OSError as exc:
            logger.error(
                f"spaCy model '{self.model_name}' not found. "
                f"Install it with: python -m spacy download {self.model_name}"
            )
            raise exc

    def _normalize_lemma(self, token) -> str:
        """
        Normalize a token lemma defensively.
        """
        lemma = (token.lemma_ or "").strip().lower()

        if not lemma:
            return ""

        if lemma == "-pron-":
            lemma = token.text.lower().strip()

        return lemma

    def _extract_lemmas(self, doc: spacy.tokens.Doc, idea_units_only: bool = False) -> Set[str]:
        """
        Extract normalized lemmas from a spaCy Doc.

        If idea_units_only=True, only retain content-bearing tokens.
        """
        lemmas = set()

        for token in doc:
            if token.is_space or token.is_punct:
                continue

            if token.like_num:
                continue

            lemma = self._normalize_lemma(token)
            if not lemma:
                continue

            if idea_units_only:
                if token.pos_ not in self.content_pos:
                    continue
                if self.remove_stopwords_for_ideas and token.is_stop:
                    continue

            lemmas.add(lemma)

        return lemmas

    def _compute_recall(self, source: Set[str], target: Set[str]) -> float:
        """
        Recall = matched source units / total source units
        Used because EIT scoring is about how much of the stimulus was preserved.
        """
        if not source:
            return 0.0
        return len(source & target) / len(source)

    def _compute_precision(self, source: Set[str], target: Set[str]) -> float:
        """
        Precision = matched target units / total target units
        Useful for analysis, even if recall is the main scoring signal.
        """
        if not target:
            return 0.0
        return len(source & target) / len(target)

    def extract_features(self, clean_stimulus: str, clean_response: str) -> Dict[str, float]:
        """
        Extract linguistic overlap features between cleaned stimulus and response.
        """
        if not clean_stimulus or not clean_response:
            return {
                "lemma_recall": 0.0,
                "idea_unit_recall": 0.0,
                "lemma_precision": 0.0,
                "idea_unit_precision": 0.0,
                "stimulus_lemma_count": 0,
                "response_lemma_count": 0,
                "stimulus_idea_count": 0,
                "response_idea_count": 0,
            }

        doc_stim = self.nlp(clean_stimulus)
        doc_resp = self.nlp(clean_response)

        stim_lemmas = self._extract_lemmas(doc_stim, idea_units_only=False)
        resp_lemmas = self._extract_lemmas(doc_resp, idea_units_only=False)

        stim_ideas = self._extract_lemmas(doc_stim, idea_units_only=True)
        resp_ideas = self._extract_lemmas(doc_resp, idea_units_only=True)

        lemma_recall = self._compute_recall(stim_lemmas, resp_lemmas)
        idea_unit_recall = self._compute_recall(stim_ideas, resp_ideas)

        lemma_precision = self._compute_precision(stim_lemmas, resp_lemmas)
        idea_unit_precision = self._compute_precision(stim_ideas, resp_ideas)

        return {
            "lemma_recall": round(lemma_recall, 4),
            "idea_unit_recall": round(idea_unit_recall, 4),
            "lemma_precision": round(lemma_precision, 4),
            "idea_unit_precision": round(idea_unit_precision, 4),
            "stimulus_lemma_count": len(stim_lemmas),
            "response_lemma_count": len(resp_lemmas),
            "stimulus_idea_count": len(stim_ideas),
            "response_idea_count": len(resp_ideas),
        }