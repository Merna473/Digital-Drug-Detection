import os
import hashlib
from pathlib import Path
import kagglehub
import pandas as pd

DATASET_HANDLE = "mohamedgharzouli/digital-drug-music"
dataset_path = kagglehub.dataset_download(DATASET_HANDLE)
DATASET_PATH = Path(dataset_path)

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
INPUT_CSV = PROCESSED_DIR / "audio_inspection.csv"
OUTPUT_CSV = PROCESSED_DIR / "dataset_validation.csv"


def _resolve_audio_path(raw_path_value: object) -> Path:
    """Resolve relative dataset paths against kagglehub DATASET_PATH while supporting full paths."""
    raw_str = "" if pd.isna(raw_path_value) else str(raw_path_value).strip()
    if not raw_str:
        return Path("")
    candidate = Path(raw_str)
    if candidate.is_absolute():
        return candidate
    return DATASET_PATH / candidate


def _compute_sha256(file_path: str) -> str:
    """Compute SHA-256 hash for an audio file to detect exact duplicates."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def _preview_duplicates(df: pd.DataFrame, limit: int = 5) -> None:
    """Print sample duplicate files alongside their matching original file to verify manually."""
    dups = df[df["is_duplicate"] == True]
    print(f"\n========== DUPLICATE VERIFICATION SAMPLE (showing up to {limit} of {len(dups)}) ==========")
    if dups.empty:
        print("No duplicate files found.")
        return

    originals = df[(df["is_duplicate"] == False) & (df["sha256"] != "")]
    for _, row in dups.head(limit).iterrows():
        matched = originals[originals["sha256"] == row["sha256"]]
        orig_name = matched.iloc[0]["file_name"] if not matched.empty else "N/A"
        orig_path = matched.iloc[0]["file_path"] if not matched.empty else "N/A"
        print(
            f"- Duplicate: {row['file_name']} ({row['file_path']}) "
            f"| Original: {orig_name} ({orig_path}) "
            f"| SHA-256: {str(row['sha256'])[:16]}..."
        )


def run_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main validation function  :
1. Verify label validity.
2. Compute SHA-256 hash for each accepted file and detect exact duplicates.
3. Update status and reject_reason, and add sha256 and is_duplicate columns.
    """
    print("\n========== DATASET VALIDATION & DUPLICATE CHECK ==========")
    print(f"Total incoming rows: {len(df)}")

    df = df.copy()

    # Initialize agreed columns
    df["sha256"] = ""
    df["is_duplicate"] = False

    if "status" not in df.columns:
        df["status"] = "accepted"
    if "reject_reason" not in df.columns:
        df["reject_reason"] = ""

    df["status"] = df["status"].fillna("accepted").astype(str)
    df["reject_reason"] = df["reject_reason"].fillna("").astype(str)

    seen_hashes = {}

    # Allowed label values (supporting both string DD/NDD and numeric 0/1)
    valid_labels = {"DD", "NDD", 0, 1, "0", "1"}

    skipped_count = 0
    invalid_label_count = 0
    missing_file_count = 0
    hash_error_count = 0
    duplicate_count = 0
    accepted_final_count = 0

    for idx, row in df.iterrows():
        # Skip files already rejected during exploration
        if row.get("status") != "accepted":
            skipped_count += 1
            continue

        # 1. Label Verification
        label_val = row.get("label")
        normalized_label = label_val.strip().upper() if isinstance(label_val, str) else label_val
        if pd.isna(label_val) or (label_val not in valid_labels and normalized_label not in valid_labels):
            df.at[idx, "status"] = "rejected"
            df.at[idx, "reject_reason"] = "unverified_label"
            invalid_label_count += 1
            continue

        # 2. Check file existence and compute SHA-256 hash
        resolved_path = _resolve_audio_path(row.get("file_path", ""))
        file_path = str(resolved_path)

        if not file_path or not os.path.isfile(file_path):
            df.at[idx, "status"] = "rejected"
            df.at[idx, "reject_reason"] = "file_not_found"
            missing_file_count += 1
            continue

        try:
            file_hash = _compute_sha256(file_path)
            df.at[idx, "sha256"] = file_hash
        except Exception as e:
            df.at[idx, "status"] = "rejected"
            df.at[idx, "reject_reason"] = f"hash_error: {e}"
            hash_error_count += 1
            continue

        # 3. Exact Duplicate Removal
        if file_hash in seen_hashes:
            df.at[idx, "is_duplicate"] = True
            df.at[idx, "status"] = "rejected"
            df.at[idx, "reject_reason"] = "exact_duplicate"
            duplicate_count += 1
        else:
            seen_hashes[file_hash] = row.get("file_name", "")
            accepted_final_count += 1

    print("\nValidation Summary:")
    print(f"- Skipped (already rejected earlier): {skipped_count}")
    print(f"- Rejected due to invalid label: {invalid_label_count}")
    print(f"- Rejected due to missing file: {missing_file_count}")
    print(f"- Rejected due to hash error: {hash_error_count}")
    print(f"- Rejected due to exact duplicate: {duplicate_count}")
    print(f"- Unique hashes computed: {len(seen_hashes)}")
    print(f"- Accepted files after validation: {accepted_final_count}")
    print(f"- Total rejected files in table: {(df['status'] == 'rejected').sum()}")

    return df


if __name__ == "__main__":
    print(f"Dataset path: {dataset_path}")
    print(f"Processed folder: {PROCESSED_DIR}")

    if INPUT_CSV.is_file():
        print(f"\nLoading inspection data from: {INPUT_CSV}")
        inspection_df = pd.read_csv(INPUT_CSV)
        result_df = run_validation(inspection_df)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved validation output to: {OUTPUT_CSV}")
        print(f"Output columns: {result_df.columns.tolist()}")
        _preview_duplicates(result_df, limit=5)
    else:
        print(f"\nInspection CSV not found at {INPUT_CSV}. Running quick sanity test...")
        sample_files = sorted(Path(dataset_path).rglob("*.mp3"))[:2]

        if len(sample_files) >= 2:
            test_data = pd.DataFrame([
                {
                    "file_name": sample_files[0].name,
                    "file_path": str(sample_files[0].relative_to(DATASET_PATH)),
                    "folder_name": sample_files[0].parent.name,
                    "label": "DD",
                    "duration_sec": 10.0,
                    "channels": 2,
                    "orig_sr": 44100,
                    "status": "accepted",
                    "reject_reason": "ok",
                },
                {
                    # Intentional duplicate of the first file for testing
                    "file_name": "copy_" + sample_files[0].name,
                    "file_path": str(sample_files[0].relative_to(DATASET_PATH)),
                    "folder_name": sample_files[0].parent.name,
                    "label": "DD",
                    "duration_sec": 10.0,
                    "channels": 2,
                    "orig_sr": 44100,
                    "status": "accepted",
                    "reject_reason": "ok",
                },
                {
                    "file_name": sample_files[1].name,
                    "file_path": str(sample_files[1].relative_to(DATASET_PATH)),
                    "folder_name": sample_files[1].parent.name,
                    "label": "NDD",
                    "duration_sec": 12.0,
                    "channels": 2,
                    "orig_sr": 44100,
                    "status": "accepted",
                    "reject_reason": "ok",
                },
            ])

            result_df = run_validation(test_data)
            print("\nValidation test result:")
            print(result_df[["file_name", "label", "status", "reject_reason", "is_duplicate", "sha256"]])
        else:
            print("No MP3 files found in dataset path for testing.")