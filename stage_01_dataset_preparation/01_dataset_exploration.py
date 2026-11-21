
import os
import pandas as pd
import librosa
from pathlib import Path

# ==========================================
# PATHS
# ==========================================

DATASET_PATH = Path(
    r"C:\Users\maisa\.cache\kagglehub\datasets\mohamedgharzouli\digital-drug-music\versions\2"
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

AUDIO_EXTENSIONS = {".mp3"}

# ==========================================
# AUDIO INFORMATION
# ==========================================

def get_audio_info(file_path):

    audio, sample_rate = librosa.load(
        str(file_path),
        sr=None,
        mono=False
    )

    if audio.ndim == 1:
        channels = 1
        samples = len(audio)
    else:
        channels = audio.shape[0]
        samples = audio.shape[-1]

    duration = samples / sample_rate

    return duration, channels, sample_rate


# ==========================================
# TASK 1: AUDIO INSPECTION
# ==========================================

def run_exploration():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    records = []

    for label in ["DD", "NDD"]:

        folder_path = DATASET_PATH / label

        if not folder_path.exists():
            print(f"Warning: {label} folder not found")
            continue

        for root, dirs, files in os.walk(folder_path):

            for file in files:

                file_path = Path(root) / file

                if file_path.suffix.lower() not in AUDIO_EXTENSIONS:
                    continue

                status = "accepted"
                reject_reason = ""

                duration = None
                channels = None
                sample_rate = None

                try:

                    duration, channels, sample_rate = get_audio_info(
                        file_path
                    )

                    if duration <= 0:
                        status = "rejected"
                        reject_reason = "Invalid audio duration"

                except Exception as error:

                    status = "rejected"
                    reject_reason = f"Audio reading error: {error}"

                records.append({

                    "file_name": file,
                    "file_path": str(file_path),
                    "folder_name": file_path.parent.name,
                    "label": label,
                    "duration_sec": duration,
                    "channels": channels,
                    "orig_sr": sample_rate,
                    "status": status,
                    "reject_reason": reject_reason

                })

    columns = [
        "file_name",
        "file_path",
        "folder_name",
        "label",
        "duration_sec",
        "channels",
        "orig_sr",
        "status",
        "reject_reason"
    ]

    df = pd.DataFrame(records, columns=columns)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "audio_inspection.csv"

    df.to_csv(output_file, index=False)

    # ======================================
    # SUMMARY
    # ======================================

    print("\n========== AUDIO INSPECTION SUMMARY ==========")

    print("Total audio files:", len(df))

    print("DD files:", (df["label"] == "DD").sum())

    print("NDD files:", (df["label"] == "NDD").sum())

    print("Accepted files:", (df["status"] == "accepted").sum())

    print("Rejected files:", (df["status"] == "rejected").sum())

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print("\nSample of inspected files:")
    print(df.head(10).to_string(index=False))

    print("\nResults saved to:", output_file)

    return df


if __name__ == "__main__":
    run_exploration()
