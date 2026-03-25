import yaml
import logging

from src.data.data_loader import EITDataLoader
from src.preprocessing.preprocessor import EITPreprocessor
from src.features.feature_extractor import MasterFeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_sanity_check():
    try:
        with open("configs/config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        data_loader = EITDataLoader(config)
        preprocessor = EITPreprocessor(config)
        feature_extractor = MasterFeatureExtractor(config)

        tuning_df, _ = data_loader.load_data()

        if tuning_df.empty:
            logger.error("Data failed to load. Check your data/raw/ folder.")
            return

        sample_data = tuning_df.head(3)

        print("\n" + "=" * 60)
        print("🚀 RUNNING AI INFERENCE SANITY TEST")
        print("=" * 60)

        pairs_to_score = []
        row_metadata = []

        for idx, row in sample_data.iterrows():
            clean_stim, clean_resp, early_gate = preprocessor.preprocess_pair(
                row["stimulus"], row["transcription"]
            )

            print(f"\n--- Row {idx} ---")
            print(f"Original Stimulus: {row['stimulus']}")
            print(f"Original Response: {row['transcription']}")
            print(f"Cleaned Stimulus : {clean_stim}")
            print(f"Cleaned Response : {clean_resp}")
            print(f"Early Gate Score : {early_gate}")

            if early_gate == -1:
                pairs_to_score.append((clean_stim, clean_resp))
                row_metadata.append(idx)

        if pairs_to_score:
            print("\n" + "=" * 60)
            print("🧠 EXTRACTING FEATURES")
            print("=" * 60)

            features_batch = feature_extractor.extract_features_batch(pairs_to_score)

            for row_idx, (stim, resp), features in zip(row_metadata, pairs_to_score, features_batch):
                print(f"\n--- Features for Row {row_idx} ---")
                print(f"Target : {stim}")
                print(f"Student: {resp}")
                print("Extracted Features:")
                for key, value in features.items():
                    print(f"  {key}: {value}")
        else:
            print("\nNo rows required AI feature extraction. All were handled by the early gate.")

    except Exception as e:
        logger.exception(f"Sanity check failed: {e}")


if __name__ == "__main__":
    run_sanity_check()