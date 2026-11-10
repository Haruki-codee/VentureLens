from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

FEATURES = [
    "sector", "stage", "city", "amount_usd", "investor_count", "startup_age",
    "prior_round_count", "prior_funding_total", "days_since_prev_round", "prev_amount_usd"
]
CATEGORICAL = ["sector", "stage", "city"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]

def build_training_data(df: pd.DataFrame, horizon_days: int = 730):
    data = df.copy().sort_values(["company_name", "announced_date"]).reset_index(drop=True)
    data["announced_date"] = pd.to_datetime(data["announced_date"])
    data["founded_year"] = pd.to_numeric(data.get("founded_year"), errors="coerce")
    rows = []
    max_date = data["announced_date"].max()
    for company, group in data.groupby("company_name", sort=False):
        group = group.sort_values("announced_date").reset_index(drop=True)
        dates = group["announced_date"].tolist()
        amounts = pd.to_numeric(group["amount_usd"], errors="coerce").tolist()
        for i, row in group.iterrows():
            index_date = dates[i]
            if pd.isna(index_date) or (max_date - index_date).days < horizon_days:
                continue
            next_dates = [d for d in dates[i+1:] if d > index_date]
            next_round = min(next_dates) if next_dates else None
            label = int(next_round is not None and (next_round - index_date).days <= horizon_days)
            previous_amounts = [a for a in amounts[:i] if pd.notna(a)]
            previous_dates = [d for d in dates[:i] if d < index_date]
            founded_year = row.get("founded_year")
            startup_age = max(0, index_date.year - founded_year) if pd.notna(founded_year) else np.nan
            prev_date = max(previous_dates) if previous_dates else None
            rows.append({
                "announced_date": index_date,
                "company_name": company,
                "sector": row.get("sector", "Unknown"),
                "stage": row.get("stage", "Unknown"),
                "city": row.get("city", "Unknown"),
                "amount_usd": row.get("amount_usd", np.nan),
                "investor_count": row.get("investor_count", np.nan),
                "startup_age": startup_age,
                "prior_round_count": i,
                "prior_funding_total": float(np.nansum(previous_amounts)) if previous_amounts else 0.0,
                "days_since_prev_round": (index_date - prev_date).days if prev_date else 0,
                "prev_amount_usd": previous_amounts[-1] if previous_amounts else np.nan,
                "target_next_round_24m": label,
            })
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No eligible rows. A full 24-month follow-up window is required for each training example.")
    return result.sort_values("announced_date").reset_index(drop=True)

def _make_pipeline(model_name="logistic_regression"):
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2))
    ])
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    preprocess = ColumnTransformer([
        ("cat", categorical_pipe, CATEGORICAL),
        ("num", numeric_pipe, NUMERIC)
    ])
    if model_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=350, min_samples_leaf=3, class_weight="balanced",
            random_state=42, n_jobs=-1
        )
    else:
        estimator = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    return Pipeline([("preprocess", preprocess), ("model", estimator)])

def train_and_save(df, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    training = build_training_data(df)
    unique_dates = np.sort(training["announced_date"].unique())
    if len(unique_dates) < 3:
        raise ValueError("Not enough distinct dates for chronological train/test split.")
    split_date = unique_dates[max(1, int(len(unique_dates) * 0.8))]
    train = training[training["announced_date"] < split_date]
    test = training[training["announced_date"] >= split_date]
    if train.empty or test.empty or train["target_next_round_24m"].nunique() < 2:
        raise ValueError("Chronological split does not contain enough classes. More labelled history is needed.")
    X_train, y_train = train[FEATURES], train["target_next_round_24m"]
    X_test, y_test = test[FEATURES], test["target_next_round_24m"]
    candidates = []
    for name in ["logistic_regression", "random_forest"]:
        pipe = _make_pipeline(name)
        pipe.fit(X_train, y_train)
        p = pipe.predict_proba(X_test)[:, 1]
        pred = (p >= 0.5).astype(int)
        metrics = {
            "model_name": name,
            "roc_auc": float(roc_auc_score(y_test, p)) if y_test.nunique() > 1 else None,
            "pr_auc": float(average_precision_score(y_test, p)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "train_rows": int(len(train)), "test_rows": int(len(test)),
            "test_period": f"{test['announced_date'].min().date()} to {test['announced_date'].max().date()}",
            "label_definition": "another recorded funding round within 730 days after index round",
            "split_date": str(pd.Timestamp(split_date).date()),
        }
        candidates.append((metrics["pr_auc"], pipe, metrics))
    _, best_model, best_metrics = max(candidates, key=lambda x: x[0])
    joblib.dump(best_model, output_dir / "funding_activity_model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    (output_dir / "feature_columns.json").write_text(json.dumps(FEATURES, indent=2), encoding="utf-8")
    training.to_csv(output_dir.parent / "reports" / "labelled_training_data.csv", index=False)
    return best_metrics

def load_model_artifacts(model_dir: Path):
    model_path = model_dir / "funding_activity_model.joblib"
    metrics_path = model_dir / "metrics.json"
    features_path = model_dir / "feature_columns.json"
    if not (model_path.exists() and metrics_path.exists() and features_path.exists()):
        return None
    return (
        joblib.load(model_path),
        json.loads(metrics_path.read_text(encoding="utf-8")),
        json.loads(features_path.read_text(encoding="utf-8"))
    )
