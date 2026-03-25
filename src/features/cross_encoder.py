# import logging
# from typing import Dict, List, Tuple

# import torch
# from sentence_transformers.cross_encoder import CrossEncoder

# logger = logging.getLogger(__name__)


# class CrossEncoderFeatureExtractor:
#     def __init__(self, config: dict):
#         """
#         Cross-Encoder semantic similarity extractor for AutoEIT.
#         Evaluates stimulus-response pairs jointly for deeper semantic matching.
#         """
#         models_config = config.get("models", {})
#         scoring_config = config.get("scoring_engine", {})

#         self.model_name = models_config.get(
#             "cross_encoder",
#             "cross-encoder/stsb-xlm-r-multilingual",
#         )
#         self.batch_size = models_config.get("batch_size", 16)
#         self.device = self._determine_device(models_config.get("device", "cpu"))

#         # Future-proof score normalization strategy
#         self.score_normalization = scoring_config.get(
#             "cross_encoder_score_normalization",
#             "clamp"
#         )

#         self.model = self._load_model()

#     def _determine_device(self, config_device: str) -> str:
#         """
#         Resolve the requested device safely.
#         Supports CPU, CUDA, and Apple MPS with graceful fallback.
#         """
#         config_device = str(config_device).lower()

#         if config_device == "cuda":
#             if torch.cuda.is_available():
#                 return "cuda"
#             logger.warning("CUDA requested but not available. Falling back to CPU.")
#             return "cpu"

#         if config_device == "mps":
#             if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#                 return "mps"
#             logger.warning("MPS requested but not available. Falling back to CPU.")
#             return "cpu"

#         return "cpu" if config_device not in {"cpu", "cuda", "mps"} else config_device

#     def _load_model(self) -> CrossEncoder:
#         """
#         Load the configured CrossEncoder model.
#         """
#         try:
#             logger.info(f"Loading Cross-Encoder: {self.model_name} on {self.device.upper()}")
#             return CrossEncoder(self.model_name, device=self.device)
#         except Exception as exc:
#             logger.error(
#                 "Failed to load Cross-Encoder model. "
#                 "Check that sentence-transformers and torch are installed."
#             )
#             raise exc

#     def _normalize_score(self, score: float) -> float:
#         """
#         Normalize raw Cross-Encoder score into a stable [0, 1] range for downstream scoring.
#         """
#         if self.score_normalization == "clamp":
#             return max(0.0, min(1.0, score))

#         # Fallback to clamp if unknown strategy is provided
#         logger.warning(
#             f"Unknown cross-encoder normalization strategy '{self.score_normalization}'. "
#             "Falling back to 'clamp'."
#         )
#         return max(0.0, min(1.0, score))

#     def compute_similarity(self, clean_stimulus: str, clean_response: str) -> float:
#         """
#         Compute deep semantic similarity between a single stimulus and response pair.
#         """
#         if not clean_stimulus or not clean_response:
#             return 0.0

#         raw_score = self.model.predict(
#             [[clean_stimulus, clean_response]],
#             batch_size=1,
#             show_progress_bar=False
#         )[0]

#         normalized_score = self._normalize_score(float(raw_score))
#         return round(normalized_score, 4)

#     def extract_features(self, clean_stimulus: str, clean_response: str) -> Dict[str, float]:
#         """
#         Extract Cross-Encoder similarity feature.
#         """
#         return {
#             "cross_encoder_similarity": self.compute_similarity(clean_stimulus, clean_response)
#         }

#     def compute_batch_similarity(self, pairs: List[Tuple[str, str]]) -> List[float]:
#         """
#         Batch semantic similarity computation for multiple stimulus-response pairs.
#         Preserves original order and safely handles empty pairs.
#         """
#         if not pairs:
#             return []

#         scores = [0.0] * len(pairs)
#         valid_pairs = []
#         valid_indices = []

#         for idx, (stimulus, response) in enumerate(pairs):
#             if stimulus and response:
#                 valid_pairs.append((stimulus, response))
#                 valid_indices.append(idx)

#         if not valid_pairs:
#             return scores

#         raw_scores = self.model.predict(
#             valid_pairs,
#             batch_size=self.batch_size,
#             show_progress_bar=False
#         )

#         for idx, raw_score in zip(valid_indices, raw_scores):
#             scores[idx] = round(self._normalize_score(float(raw_score)), 4)

#         return scores