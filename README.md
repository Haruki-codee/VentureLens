# VentureLens — Indian Startup Funding Intelligence

VentureLens is an educational, VC-analyst-oriented data science portfolio project. It explores Indian startup funding rounds and builds a transparent baseline model for **the probability of another recorded funding round within 24 months**.

It is not an investment recommendation engine, valuation model, profitability predictor, or fund-performance tracker.

## Features

- Market overview: annual rounds, disclosed funding by sector, stage mix, recent rounds.
- Startup explorer: company-level funding history.
- Deal-flow screener: filter recorded rounds by amount and lead investor.
- Funding-activity model: Logistic Regression and Random Forest baselines; selects the model with the higher chronological holdout PR-AUC.
- Metrics: ROC-AUC, PR-AUC, accuracy, precision, recall, F1, confusion matrix.
- Leakage-aware label construction: only rows with a complete 730-day follow-up window are eligible.
- Reproducible scripts, tests, and saved model artifacts.

## 1. Requirements

- Python 3.10 or 3.11 recommended
- Internet connection for the initial dataset download

## 2. Setup

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Download the real dataset

```bash
python scripts/download_data.py
```

This downloads the public Hugging Face dataset:

- Dataset: https://huggingface.co/datasets/fliarbi/indian-startup-funding
- Publisher: Fliar BI
- License: CC BY-NC 4.0

The raw CSV is saved to `data/raw/startup_funding.csv`. The dataset is not bundled in this ZIP; the downloader fetches the publisher's current version when you run it.

**License warning:** CC BY-NC 4.0 is non-commercial. Do not use this dataset or a project built from it commercially without obtaining the necessary rights. Keep attribution and review the dataset card before redistribution.

## 4. Launch the dashboard

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501).

## 5. Train the ML model

In a second terminal, with the virtual environment activated:

```bash
python scripts/train_model.py
```

The script creates:

- `models/funding_activity_model.joblib`
- `models/metrics.json`
- `models/feature_columns.json`
- `reports/labelled_training_data.csv`

Refresh the Streamlit page to use the model.

## Model design

### Prediction target

For each observed funding round, the target is 1 if the same company has another recorded funding round within the next 730 days; otherwise 0. A row is included only if the dataset's maximum date is at least 730 days after that row's date.

### Features

- Sector, stage, city
- Current round amount and investor count
- Startup age at the index round
- Number of previous recorded rounds
- Previous disclosed funding total
- Days since previous round
- Previous round amount

The pipeline uses imputation, one-hot encoding, and scaling inside scikit-learn's `Pipeline`, so preprocessing is fitted only on training data.

### Validation

A chronological holdout is used rather than a random split. Logistic Regression and Random Forest are compared using PR-AUC on the same holdout. Actual metrics are produced only after you download the data and train the model; no scores are pre-filled or fabricated.

### Important caveats

- A recorded subsequent round is not equivalent to startup success.
- No recorded round can mean no funding, missing news coverage, acquisition, shutdown, or other outcomes.
- The dataset is news-derived and coverage is not uniform.
- The sample is not a complete census of Indian startups or all funding rounds.
- Recent observations are excluded from supervised training unless a full two-year outcome window is observable.
- Company name matching and duplicate reporting may affect company histories.
- This dataset does not contain fund cash flows, entry valuations, exit proceeds, or portfolio marks. Therefore, real IRR, DPI, TVPI, and MOIC cannot be computed from it.

## Project structure

```text
VentureLens/
├── app.py
├── requirements.txt
├── README.md
├── DATA_LICENSE.md
├── src/venturelens/
│   ├── __init__.py
│   ├── data.py
│   ├── analytics.py
│   └── modeling.py
├── scripts/
│   ├── download_data.py
│   └── train_model.py
├── tests/
├── data/raw/
├── models/
└── reports/
```

## Run tests

```bash
pip install pytest
$env:PYTHONPATH="src"  # PowerShell
pytest -q
```

On macOS/Linux:

```bash
PYTHONPATH=src pytest -q
```

## Suggested interview explanation

"I built VentureLens, a startup funding intelligence dashboard using a public Indian funding-round dataset. I engineered a forward-looking target—whether a company records another funding round within 24 months—and excluded examples without a complete follow-up window. I compared Logistic Regression and Random Forest using a chronological holdout and PR-AUC. The product also includes market analytics and a company explorer. I explicitly treat funding continuation as an activity proxy, not investment return or startup success."

## Disclaimer

For educational and portfolio demonstration only. Not financial advice. Verify the source data, rights, assumptions, and outputs before relying on them.
