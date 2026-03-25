import logging
from typing import Dict, List, Tuple, Optional

from .linguistic import LinguisticFeatureExtractor
from .sbert import SBERTFeatureExtractor
from .nli_scorer import NLIFeatureExtractor

logger = logging.getLogger(__name__)


class MasterFeatureExtractor:
    def __init__(self, config: dict):
        """
        Orchestrates all AutoEIT feature engines.
        Initializes only the enabled models based on config toggles.
        """
        self.config = config
        features_config = config.get("features", {})

        self.use_linguistic = features_config.get("use_lemma_overlap", True)
        self.use_sbert = features_config.get("use_sbert", True)
        self.use_nli = features_config.get("use_nli", True)

        self.linguistic_engine: Optional[LinguisticFeatureExtractor] = None
        self.sbert_engine: Optional[SBERTFeatureExtractor] = None
        self.nli_engine: Optional[NLIFeatureExtractor] = None

        logger.info("--- Initializing AI Feature Extractors ---")

        if self.use_linguistic:
            self.linguistic_engine = LinguisticFeatureExtractor(config)
        else:
            logger.info("Linguistic extraction disabled via config.")

        if self.use_sbert:
            self.sbert_engine = SBERTFeatureExtractor(config)
        else:
            logger.info("SBERT Bi-Encoder disabled via config.")

        if self.use_nli:
            self.nli_engine = NLIFeatureExtractor(config)
        else:
            logger.info("NLI semantic extractor disabled via config.")

        logger.info(f"Enabled engines: {', '.join(self.get_enabled_engines()) or 'None'}")
        logger.info("--- AI Feature Extractors Ready ---")

    def get_enabled_engines(self) -> List[str]:
        """
        Returns the list of enabled feature engines.
        """
        enabled = []
        if self.use_linguistic:
            enabled.append("linguistic")
        if self.use_sbert:
            enabled.append("sbert")
        if self.use_nli:
            enabled.append("nli")
        return enabled

    def _initialize_feature_dict(self) -> Dict[str, float]:
        """
        Provides a stable feature schema for downstream components.
        """
        features: Dict[str, float] = {}

        if self.use_linguistic:
            features.update({
                "lemma_recall": 0.0,
                "idea_unit_recall": 0.0,
                "lemma_precision": 0.0,
                "idea_unit_precision": 0.0,
                "stimulus_lemma_count": 0,
                "response_lemma_count": 0,
                "stimulus_idea_count": 0,
                "response_idea_count": 0,
            })

        if self.use_sbert:
            features["sbert_similarity"] = 0.0

        if self.use_nli:
            features.update({
                "nli_entailment": 0.0,
                "nli_neutral": 0.0,
                "nli_contradiction": 0.0,
                "nli_margin": 0.0,
            })

        return features

    def extract_features(self, clean_stimulus: str, clean_response: str) -> Dict[str, float]:
        """
        Extract features for a single stimulus-response pair.
        """
        combined_features = self._initialize_feature_dict()

        if not clean_stimulus or not clean_response:
            return combined_features

        if self.linguistic_engine is not None:
            combined_features.update(
                self.linguistic_engine.extract_features(clean_stimulus, clean_response)
            )

        if self.sbert_engine is not None:
            combined_features.update(
                self.sbert_engine.extract_features(clean_stimulus, clean_response)
            )

        if self.nli_engine is not None:
            combined_features.update(
                self.nli_engine.extract_features(clean_stimulus, clean_response)
            )

        return combined_features

    def extract_features_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        """
        Extract features for a batch of stimulus-response pairs.
        Uses batched inference where available.
        """
        if not pairs:
            return []

        logger.info(f"Extracting features for {len(pairs)} pairs...")

        batch_results = [self._initialize_feature_dict() for _ in range(len(pairs))]

        if self.linguistic_engine is not None:
            logger.info("Running spaCy linguistic analysis...")
            for idx, (stimulus, response) in enumerate(pairs):
                if stimulus and response:
                    batch_results[idx].update(
                        self.linguistic_engine.extract_features(stimulus, response)
                    )

        if self.sbert_engine is not None:
            logger.info("Running SBERT semantic batching...")
            sbert_scores = self.sbert_engine.compute_batch_similarity(pairs)
            for idx, score in enumerate(sbert_scores):
                batch_results[idx]["sbert_similarity"] = score

        if self.nli_engine is not None:
            logger.info("Running NLI semantic inference...")
            nli_features = self.nli_engine.compute_batch_features(pairs)
            for idx, feature_dict in enumerate(nli_features):
                batch_results[idx].update(feature_dict)

        return batch_results