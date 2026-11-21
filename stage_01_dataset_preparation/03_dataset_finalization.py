import re
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
INPUT_CSV = PROCESSED_DIR / "dataset_validation.csv"
OUTPUT_CSV = PROCESSED_DIR / "dataset_manifest.csv"


def _extract_source_key(file_name: object, folder_name: object, label: object) -> str:
    """
    Derive a normalized source/group key from file_name and label/folder
    so related recordings/tracks stay in the same group and avoid data leakage.
    """
    name_str = "" if pd.isna(file_name) else str(file_name).strip()
    stem = Path(name_str).stem.lower()

    # Remove trailing copy/part/parenthesis markers like " (1)", "_copy", "-1"
    cleaned = re.sub(r"\s*\(\d+\)\s*$", "", stem)
    cleaned = re.sub(r"[_\-\s]*(copy|dup|duplicate|\d+)$", "", cleaned).strip()
    if not cleaned:
        cleaned = stem if stem else "unknown"

    prefix_val = "" if pd.isna(label) else str(label).strip().upper()
    if not prefix_val and not pd.isna(folder_name):
        prefix_val = str(folder_name).strip().upper()

    return f"{prefix_val}_{cleaned}" if prefix_val else cleaned


def run_finalization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main finalization function for Stage 1 (03_dataset_finalization.py):
    1. Assign group_id to prevent data leakage across train/val/test splits.
    2. Print summary of accepted/rejected rows and group counts.
    """
    print("\n========== GROUP IDENTIFICATION & MANIFEST FINALIZATION ==========")
    print(f"Total incoming rows: {len(df)}")

    df = df.copy()

    group_map = {}
    group_ids = []

    for _, row in df.iterrows():
        source_key = _extract_source_key(
            row.get("file_name", ""),
            row.get("folder_name", ""),
            row.get("label", ""),
        )
        if source_key not in group_map:
            group_map[source_key] = f"GRP_{len(group_map) + 1:05d}"
        group_ids.append(group_map[source_key])

    df["group_id"] = group_ids

    accepted_df = df[df.get("status", "") == "accepted"]
    print("\nFinalization Summary:")
    print(f"- Total rows in manifest: {len(df)}")
    print(f"- Accepted files ready for next stage: {len(accepted_df)}")
    print(f"- Rejected files in manifest: {(df.get('status', '') != 'accepted').sum()}")
    print(f"- Total unique groups identified: {df['group_id'].nunique()}")
    if not accepted_df.empty:
        print(f"- Unique groups in accepted files: {accepted_df['group_id'].nunique()}")
        print("\nAccepted files by label:")
        print(accepted_df["label"].value_counts())

    return df


if __name__ == "__main__":
    print(f"Processed folder: {PROCESSED_DIR}")

    if not INPUT_CSV.is_file():
        print(f"Validation CSV not found at: {INPUT_CSV}")
    else:
        print(f"\nLoading validation data from: {INPUT_CSV}")
        validation_df = pd.read_csv(INPUT_CSV)
        manifest_df = run_finalization(validation_df)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        manifest_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved final manifest to: {OUTPUT_CSV}")
        print(f"Output columns: {manifest_df.columns.tolist()}")