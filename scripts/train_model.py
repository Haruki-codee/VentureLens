from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from venturelens.data import load_local_data
from venturelens.analytics import clean_funding_data
from venturelens.modeling import train_and_save

def main():
    df = clean_funding_data(load_local_data())
    metrics = train_and_save(df, ROOT / "models")
    print("Training complete. Chronological holdout metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
