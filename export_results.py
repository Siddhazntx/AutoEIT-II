import pandas as pd
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def export_to_excel():
    input_path = Path("data/processed/holdout_predictions.csv")
    output_path = Path("data/processed/Final_AI_Grading_Report.xlsx")

    if not input_path.exists():
        logger.error(f"Could not find {input_path}. Did you run the pipeline first?")
        return

   
    logger.info(f"Loading results from {input_path}...")
    df = pd.DataFrame()
    df = pd.read_csv(input_path)

    
    preferred_order = [
        "participant_id", 
        "stimulus", 
        "transcription", 
        "heuristic_pred", 
        "early_gate_score",
        "heuristic_raw_score"
    ]
    
    
    final_cols = [c for c in preferred_order if c in df.columns]
    export_df = df[final_cols].copy()

    
    export_df.columns = [
        "ID", 
        "Target Sentence", 
        "Student Response", 
        "AI Final Grade (0-4)", 
        "Early Gate Hit",
        "Confidence Score (0-1)"
    ]

    
    logger.info(f"Exporting to {output_path}...")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        export_df.to_excel(writer, index=False, sheet_name='AI Predictions')
        
        
        worksheet = writer.sheets['AI Predictions']
        for i, col in enumerate(export_df.columns):
            column_len = max(export_df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.column_dimensions[chr(65 + i)].width = min(column_len, 50)

    print("\n" + "="*50)
    print(f" SUCCESS! AI results are ready.")
    print(f"Location: {output_path}")
    print("="*50 + "\n")

if __name__ == "__main__":
    export_to_excel()