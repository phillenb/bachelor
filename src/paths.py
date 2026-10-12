from pathlib import Path

# projectroot
ROOT_DIR = Path(__file__).resolve().parent.parent

# datafolder
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTENSITY_MATRICES_DATA_DIR = DATA_DIR / "intensity_matrices"