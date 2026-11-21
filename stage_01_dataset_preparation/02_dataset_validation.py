
import pandas as pd
import hashlib
from pathlib import Path

# Dataset path
DATASET_PATH = Path(
    r"C:\Users\maisa\.cache\kagglehub\datasets\mohamedgharzouli\digital-drug-music\versions\2"
)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "data" / "processed" / "audio_inspection.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "dataset_validation.csv"


def calculate_sha256(file_path):
    """Calculate SHA256 hash for an audio file."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def run_validation():

    # Read Audio Inspection DataFrame
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    # Fix column data types
    df["reject_reason"] = df["reject_reason"].fillna("").astype(str)
    df["status"] = df["status"].fillna("accepted").astype(str)

    # Add required columns
    df["sha256"] = ""
    df["is_duplicate"] = False

    # Validate every audio file
    for index, row in df.iterrows():

        file_path = Path(row["file_path"])
        label = str(row["label"]).strip().upper()

        reasons = []

        # Check label
        if label not in ["DD", "NDD"]:
            reasons.append("Invalid label")

        # Check folder and label consistency
        try:
            relative_path = file_path.relative_to(DATASET_PATH)
            actual_folder = relative_path.parts[0].upper()

            if actual_folder != label:
                reasons.append("Label does not match folder")

        except (ValueError, IndexError):
            reasons.append("File path is outside dataset folders")

        # Check file existence and calculate hash
        if not file_path.exists():
            reasons.append("File not found")

        else:
            try:
                file_hash = calculate_sha256(file_path)
                df.at[index, "sha256"] = file_hash

            except Exception as error:
                reasons.append(f"Hash calculation failed: {error}")

        # Update status and reject reason
        if reasons:
            df.at[index, "status"] = "rejected"
            df.at[index, "reject_reason"] = "; ".join(reasons)

    # Detect duplicate files
    valid_hashes = df[
        df["sha256"] != ""
    ].groupby("sha256").groups

    for file_hash, indexes in valid_hashes.items():

        indexes = list(indexes)

        if len(indexes) > 1:

            # Keep the first occurrence
            for duplicate_index in indexes[1:]:

                df.at[duplicate_index, "is_duplicate"] = True
                df.at[duplicate_index, "status"] = "rejected"

                old_reason = df.at[duplicate_index, "reject_reason"]

                duplicate_reason = "Duplicate audio file"

                if str(old_reason).strip() == "":
                    df.at[duplicate_index, "reject_reason"] = duplicate_reason
                else:
                    df.at[duplicate_index, "reject_reason"] = (
                        str(old_reason) + "; " + duplicate_reason
                    )

    # Save result
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    # Terminal summary
    print("\n========== LABEL VERIFICATION SUMMARY ==========")
    print(f"Total files: {len(df)}")
    print(f"Valid labels: {df['label'].astype(str).str.strip().str.upper().isin(['DD', 'NDD']).sum()}")
    print(f"Duplicates: {df['is_duplicate'].sum()}")
    print(f"Accepted files: {(df['status'] == 'accepted').sum()}")
    print(f"Rejected files: {(df['status'] == 'rejected').sum()}")

    print("\nOutput columns:")
    print(df.columns.tolist())

    print(f"\nResults saved to: {OUTPUT_FILE}")

    return df


if __name__ == "__main__":
    run_validation()
