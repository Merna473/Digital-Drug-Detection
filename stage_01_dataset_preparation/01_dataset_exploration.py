
import os
import kagglehub
import pandas as pd
import librosa
from pathlib import Path

# ==========================================
# Paths and Settings
# ==========================================

DATASET_HANDLE = "mohamedgharzouli/digital-drug-music"

dataset_path = kagglehub.dataset_download(DATASET_HANDLE)

DATASET_PATH = Path(dataset_path)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

APPROVED_LABELS = {"DD", "NDD"}

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


# ==========================================
# Helper Function
# ==========================================

def add_reject_reason(current_reason, new_reason):
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
        return None

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
    print(f"DD files: {(df['label'] == 'DD').sum()}")
    print(f"NDD files: {(df['label'] == 'NDD').sum()}")
    print(f"Accepted files: {(df['status'] == 'accepted').sum()}")
    print(f"Rejected files: {(df['status'] == 'rejected').sum()}")

    print("\nFiles by label:")
    print(df["label"].value_counts())

    print(f"\nInspection file saved to: {inspection_file}")

    return df


# ==========================================
# Task 2: Label Verification
# ==========================================

def run_validation(df):

    print("\n========== LABEL VERIFICATION ==========")

    df = df.copy()

    df["reject_reason"] = df["reject_reason"].fillna("").astype(str)
    df["status"] = df["status"].fillna("accepted").astype(str)

    for index, row in df.iterrows():

        file_path = Path(row["file_path"])

        label = str(row["label"]).strip().upper()

        reasons = []

        # Check approved labels
        if label not in APPROVED_LABELS:
            reasons.append(f"Unapproved label: {label}")

        # Check folder and label consistency
        try:
            relative_path = file_path.relative_to(DATASET_PATH)

            actual_folder = relative_path.parts[0].upper()

            if actual_folder != label:
                reasons.append("Label does not match folder name")

        except (ValueError, IndexError):
            reasons.append(
                "File is outside the expected dataset folders"
            )

        # Check file existence
        if not file_path.exists():
            reasons.append("File not found")

        # Update status and reject reason
        if reasons:
            df.at[index, "status"] = "rejected"
            df.at[index, "reject_reason"] = "; ".join(reasons)

    # Save validation results
    validation_file = OUTPUT_DIR / "dataset_validation.csv"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(validation_file, index=False)

    print("\nLabel Verification Results:")
    print(f"Total files: {len(df)}")
    print(
        f"Valid labels: "
        f"{df['label'].astype(str).str.strip().str.upper().isin(APPROVED_LABELS).sum()}"
    )
    print(f"Accepted files: {(df['status'] == 'accepted').sum()}")
    print(f"Rejected files: {(df['status'] == 'rejected').sum()}")

    print("\nOutput columns:")
    print(df.columns.tolist())

    print(f"\nValidation file saved to: {validation_file}")

    return df


# ==========================================
# Run Both Tasks
# ==========================================

if __name__ == "__main__":

    inspection_df = run_exploration()

    if inspection_df is not None:

        final_df = run_validation(inspection_df)

        print("\n========== TASK 1 AND TASK 2 COMPLETED ==========")
