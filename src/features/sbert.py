import logging
from typing import Dict, List

import torch
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


class SBERTFeatureExtractor:
    def __init__(self, config: dict):
        """
        SBERT-based semantic similarity extractor for AutoEIT.
        Produces a fast semantic similarity score between stimulus and response.
        """
        models_config = config.get("models", {})

        self.model_name = models_config.get(
            "bi_encoder",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.batch_size = models_config.get("batch_size", 16)
        self.device = self._determine_device(models_config.get("device", "cpu"))

        self.model = self._load_model()

    def _determine_device(self, config_device: str) -> str:
        """
        Resolve the requested device safely.
        Supports CPU, CUDA, and Apple MPS with graceful fallback.
        """
        config_device = str(config_device).lower()

        if config_device == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"

        if config_device == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            logger.warning("MPS requested but not available. Falling back to CPU.")
            return "cpu"

        return "cpu" if config_device not in {"cpu", "cuda", "mps"} else config_device

    def _load_model(self) -> SentenceTransformer:
        """
        Load the configured SentenceTransformer model.
        """
        try:
            logger.info(f"Loading SBERT model: {self.model_name} on {self.device.upper()}")
            return SentenceTransformer(self.model_name, device=self.device)
        except Exception as exc:
            logger.error(
                "Failed to load SBERT model. "
                "Check that sentence-transformers and torch are installed."
            )
            raise exc

    def _clamp_similarity(self, score: float) -> float:
        """
        Clamp similarity to [0, 1] for downstream scoring stability.
        """
        return max(0.0, min(1.0, score))

    def encode(self, texts: List[str]):
        """
        Encode a batch of texts into dense sentence embeddings.
        """
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def compute_similarity(self, clean_stimulus: str, clean_response: str) -> float:
        """
        Compute cosine similarity between stimulus and response embeddings.
        """
        if not clean_stimulus or not clean_response:
            return 0.0

        embeddings = self.encode([clean_stimulus, clean_response])
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

        return round(self._clamp_similarity(similarity), 4)

    def extract_features(self, clean_stimulus: str, clean_response: str) -> Dict[str, float]:
        """
        Extract SBERT semantic similarity feature.
        """
        return {
            "sbert_similarity": self.compute_similarity(clean_stimulus, clean_response)
        }

    def compute_batch_similarity(self, pairs: List[tuple[str, str]]) -> List[float]:
        """
        Compute similarity scores for multiple (stimulus, response) pairs.
        This is optional for now but useful for scaling later.
        """
        scores = []

        for stimulus, response in pairs:
            scores.append(self.compute_similarity(stimulus, response))

        return scores