import logging
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class NLIFeatureExtractor:
    def __init__(self, config: dict):
        """
        Multilingual NLI-based semantic feature extractor for AutoEIT.
        Uses premise-hypothesis inference to estimate meaning preservation.
        """
        models_config = config.get("models", {})
        features_config = config.get("features", {})

        self.model_name = models_config.get(
            "nli_model",
            "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
        )
        self.batch_size = models_config.get("batch_size", 8)
        self.max_length = models_config.get("max_length", 256)
        self.device = self._determine_device(models_config.get("device", "cpu"))
        self.premise_first = features_config.get("nli_premise_first", "stimulus")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = self._load_model()
        self.label_map = self._build_label_map()

    def _determine_device(self, config_device: str) -> str:
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

        return "cpu"

    def _load_model(self):
        try:
            logger.info(f"Loading NLI model: {self.model_name} on {self.device.upper()}")
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            model.to(self.device)
            model.eval()
            return model
        except Exception as exc:
            logger.error("Failed to load NLI model.")
            raise exc

    def _build_label_map(self) -> Dict[int, str]:
        """
        Build a normalized id -> label map from model config.
        """
        id2label = self.model.config.id2label
        normalized = {}

        for idx, label in id2label.items():
            lab = str(label).lower().strip()
            if "entail" in lab:
                normalized[int(idx)] = "entailment"
            elif "neutral" in lab:
                normalized[int(idx)] = "neutral"
            elif "contrad" in lab:
                normalized[int(idx)] = "contradiction"
            else:
                normalized[int(idx)] = lab

        return normalized

    def _prepare_pair(self, stimulus: str, response: str) -> Tuple[str, str]:
        """
        Prepare ordered premise-hypothesis pair.
        """
        if self.premise_first == "response":
            return response, stimulus
        return stimulus, response

    def compute_probabilities(self, clean_stimulus: str, clean_response: str) -> Dict[str, float]:
        """
        Compute entailment / neutral / contradiction probabilities for one pair.
        """
        if not clean_stimulus or not clean_response:
            return {
                "nli_entailment": 0.0,
                "nli_neutral": 0.0,
                "nli_contradiction": 0.0,
                "nli_margin": 0.0,
            }

        premise, hypothesis = self._prepare_pair(clean_stimulus, clean_response)

        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

        scores = {
            "entailment": 0.0,
            "neutral": 0.0,
            "contradiction": 0.0,
        }

        for idx, prob in enumerate(probs):
            label = self.label_map.get(idx, f"label_{idx}")
            if label in scores:
                scores[label] = float(prob)

        return {
            "nli_entailment": round(scores["entailment"], 4),
            "nli_neutral": round(scores["neutral"], 4),
            "nli_contradiction": round(scores["contradiction"], 4),
            "nli_margin": round(scores["entailment"] - scores["contradiction"], 4),
        }

    def extract_features(self, clean_stimulus: str, clean_response: str) -> Dict[str, float]:
        return self.compute_probabilities(clean_stimulus, clean_response)

    def compute_batch_features(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        """
        Simple batch inference for multiple pairs.
        """
        if not pairs:
            return []

        results = []

        for stimulus, response in pairs:
            results.append(self.compute_probabilities(stimulus, response))

        return results