from pathlib import Path
import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "startup_funding.csv"
REPO_ID = "fliarbi/indian-startup-funding"

def main():
    print(f"Downloading {REPO_ID} from Hugging Face...")
    dataset = load_dataset(REPO_ID)
    split_name = "train" if "train" in dataset else next(iter(dataset.keys()))
    frame = dataset[split_name].to_pandas()
    if frame.empty:
        raise RuntimeError("The dataset split downloaded but contains no records.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    print(f"Saved {len(frame):,} rows to {OUT}")
    print("Dataset attribution: Fliar BI, Indian Startup Funding Rounds 2015–2026.")
    print("License: CC BY-NC 4.0. Non-commercial use only unless separately licensed.")

if __name__ == "__main__":
    main()
