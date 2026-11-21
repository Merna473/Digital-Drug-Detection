
from pathlib import Path
import hashlib
import pandas as pd
import librosa

# ==========================================
# Paths and Settings
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = Path(
    r"C:\Users\maisa\.cache\kagglehub\datasets\mohamedgharzouli\digital-drug-music\versions\2"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

APPROVED_LABELS = {"DD", "NDD"}

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


# ==========================================
# Helper Functions
# ==========================================

def calculate_sha256(file_path):
    """Calculate SHA256 hash for an audio file."""
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


def add_reject_reason(current_reason, new_reason):
    """Add a rejection reason without deleting previous reasons."""
    if not current_reason:
        return new_reason

    return current_reason + "; " + new_reason


# ==========================================
# Task 1: Audio Inspection
# ==========================================

def run_exploration():

    print("\n========== AUDIO INSPECTION ==========")

    if not DATASET_PATH.exists():
        print("Dataset path does not exist!")
        return

    records = []

    for file_path in sorted(DATASET_PATH.rglob("*")):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        relative_path = file_path.relative_to(DATASET_PATH)

        folder_name = relative_path.parts[0] if relative_path.parts else ""

        label = folder_name

        duration_sec = None
        channels = None
        orig_sr = None

        status = "accepted"
        reject_reason = ""

        try:
            audio, sr = librosa.load(
                str(file_path),
                sr=None,
                mono=False
            )

            orig_sr = sr

            if audio.ndim == 1:
                channels = 1
            else:
                channels = audio.shape[0]

            duration_sec = len(audio) / sr

            if duration_sec <= 0:
                status = "rejected"
                reject_reason = add_reject_reason(
                    reject_reason,
                    "Invalid audio duration"
                )

        except Exception as error:
            status = "rejected"
            reject_reason = add_reject_reason(
                reject_reason,
                f"Audio inspection failed: {error}"
            )

        records.append({
            "file_name": file_path.name,
            "file_path": str(file_path),
            "folder_name": folder_name,
            "label": label,
            "duration_sec": duration_sec,
            "channels": channels,
            "orig_sr": orig_sr,
            "status": status,
            "reject_reason": reject_reason
        })

    df = pd.DataFrame(records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inspection_file = OUTPUT_DIR / "audio_inspection.csv"
    df.to_csv(inspection_file, index=False)

    print("\nAudio Inspection Results:")
    print(f"Total audio files: {len(df)}")
    print(f"Accepted files: {(df['status'] == 'accepted').sum()}")
    print(f"Rejected files: {(df['status'] == 'rejected').sum()}")

    print("\nFiles by label:")
    print(df["label"].value_counts())

    print(f"\nInspection file saved to: {inspection_file}")

    return df


# ==========================================
# Task 2: Label Verification
# ==========================================

def run_label_verification(df):

    print("\n========== LABEL VERIFICATION ==========")

    df = df.copy()

    df["sha256"] = None
    df["is_duplicate"] = False

    seen_hashes = set()

    for index, row in df.iterrows():

        file_path = Path(row["file_path"])

        # Check whether the label is approved
        if row["label"] not in APPROVED_LABELS:

            df.at[index, "status"] = "rejected"

            df.at[index, "reject_reason"] = add_reject_reason(
                row["reject_reason"],
                f"Unapproved label: {row['label']}"
            )

        # Check whether the file path matches its label
        try:
            relative_path = file_path.relative_to(DATASET_PATH)
            actual_folder = relative_path.parts[0]

            if actual_folder != row["label"]:

                df.at[index, "status"] = "rejected"

                df.at[index, "reject_reason"] = add_reject_reason(
                    df.at[index, "reject_reason"],
                    "Label does not match folder name"
                )

        except (ValueError, IndexError):

            df.at[index, "status"] = "rejected"

            df.at[index, "reject_reason"] = add_reject_reason(
                df.at[index, "reject_reason"],
                "File is outside the expected dataset folders"
            )

        # Calculate SHA256 for each file
        try:
            file_hash = calculate_sha256(file_path)
            df.at[index, "sha256"] = file_hash

            # Detect duplicate files
            if file_hash in seen_hashes:

                df.at[index, "is_duplicate"] = True
                df.at[index, "status"] = "rejected"

                df.at[index, "reject_reason"] = add_reject_reason(
                    df.at[index, "reject_reason"],
                    "Duplicate audio file"
                )

            else:
                seen_hashes.add(file_hash)

        except Exception as error:

            df.at[index, "status"] = "rejected"

            df.at[index, "reject_reason"] = add_reject_reason(
                df.at[index, "reject_reason"],
                f"SHA256 calculation failed: {error}"
            )

    # Save final validation results
    validation_file = OUTPUT_DIR / "dataset_validation.csv"

    df.to_csv(validation_file, index=False)

    print("\nLabel Verification Results:")
    print(f"Total files: {len(df)}")
    print(f"Accepted files: {(df['status'] == 'accepted').sum()}")
    print(f"Rejected files: {(df['status'] == 'rejected').sum()}")
    print(f"Duplicate files: {df['is_duplicate'].sum()}")

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print(f"\nValidation file saved to: {validation_file}")

    return df


# ==========================================
# Run Both Tasks
# ==========================================

if __name__ == "__main__":

    inspection_df = run_exploration()

    if inspection_df is not None:

        final_df = run_label_verification(inspection_df)

        print("\n========== ALL TASKS COMPLETED ==========")
