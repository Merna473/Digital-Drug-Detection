from pathlib import Path
import sys
import kagglehub
import pandas as pd

DATASET_HANDLE = "mohamedgharzouli/digital-drug-music"

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MANIFEST_CSV = PROCESSED_DIR / "dataset_manifest.csv"

# Ensure project root is importable regardless of current working directory or runner
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def get_dataset_root() -> Path:
    """Return local kagglehub dataset directory."""
    return Path(kagglehub.dataset_download(DATASET_HANDLE))


def resolve_audio_file(raw_path: object, dataset_root: Path) -> Path:
    """Resolve relative dataset file_path against dataset_root, supporting full paths as fallback."""
    raw_str = "" if pd.isna(raw_path) else str(raw_path).strip()
    if not raw_str:
        return Path("")
    candidate = Path(raw_str)
    return candidate if candidate.is_absolute() else (dataset_root / candidate)


def load_clean_dataset(
    manifest_path: Path = MANIFEST_CSV,
    verify_files: bool = True,
) -> pd.DataFrame:
    """
    Safely load ONLY the cleaned, accepted, non-duplicate records from dataset_manifest.csv
    and attach 'full_path' ready for audio loading.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    # Keep only accepted, non-duplicate rows
    clean_df = df[df["status"] == "accepted"].copy()
    if "is_duplicate" in clean_df.columns:
        clean_df = clean_df[clean_df["is_duplicate"] != True].copy()

    # Resolve full paths against dataset_root
    dataset_root = get_dataset_root()
    clean_df["full_path"] = clean_df["file_path"].apply(
        lambda p: str(resolve_audio_file(p, dataset_root))
    )

    if verify_files:
        missing = clean_df[~clean_df["full_path"].apply(lambda p: Path(p).is_file())]
        if not missing.empty:
            raise FileNotFoundError(f"Found {len(missing)} missing audio files!")

    if "sha256" in clean_df.columns:
        assert clean_df["sha256"].nunique() == len(clean_df), (
            "Duplicate hashes detected in clean dataset!"
        )

    return clean_df.reset_index(drop=True)