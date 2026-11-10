from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "startup_funding.csv"

def load_local_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(path, low_memory=False)
