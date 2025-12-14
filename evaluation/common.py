from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)


DEFAULT_DROP_COLS = {
    "window_id",
    "start_time",
    "end_time",
    "n_events",
    "n_rows",
    "split",
    "label",
}


@dataclass
class ModelArtifacts:
    dataset: str
    out_dir: str
    model_path: str
    label_encoder_path: str
    feature_cols_path: str
    metrics_path: str
    confusion_val_csv: Optional[str] = None
    confusion_test_csv: Optional[str] = None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_parquet(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet file: {path}")
    return pd.read_parquet(path)


def infer_feature_cols(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    extra_drop_cols: Optional[Sequence[str]] = None,
) -> List[str]:
    drop = set(DEFAULT_DROP_COLS)
    drop.add(label_col)
    if extra_drop_cols:
        drop.update(extra_drop_cols)

    feat_cols: List[str] = []
    for c in df.columns:
        if c in drop:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feat_cols.append(c)

    if not feat_cols:
        raise RuntimeError(
            "No numeric feature columns detected. Ensure preprocessing produced numeric features."
        )
    return feat_cols


def make_xy(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    label_col: str = "label",
) -> Tuple[np.ndarray, np.ndarray]:
    X = df.loc[:, list(feature_cols)].copy().fillna(0.0)
    y = df[label_col].astype(str).to_numpy()
    return X.to_numpy(dtype=float), y


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def save_confusion_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Sequence[str],
    out_csv: str | Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    df_cm = pd.DataFrame(cm, index=list(labels), columns=list(labels))
    out_csv = Path(out_csv)
    ensure_dir(out_csv.parent)
    df_cm.to_csv(out_csv, index=True)


def build_model(dataset: str, seed: int, model_name: Optional[str] = None) -> BaseEstimator:
    """
    Baseline defaults:
      - CASAS: LogisticRegression (good baseline on wide bag-of-events)
      - SPHERE: RandomForestClassifier (robust on low-dim continuous features)
    """
    name = (model_name or "").strip().lower()

    if name in {"rf", "random_forest", "randomforest"}:
        return RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )

    if dataset.lower() == "sphere":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )

    return LogisticRegression(
        max_iter=5000,
        solver="saga",
      #  n_jobs=-1,
        class_weight="balanced",
       # multi_class="auto",
        random_state=seed,
    )



def safe_train_val_test_split(
    df: pd.DataFrame,
    *,
    train: float,
    val: float,
    test: float,
    seed: int,
    stratify_col: Optional[str] = "label",
) -> Dict[str, pd.DataFrame]:
    """
    Split with stratification when possible; falls back gracefully if stratify fails
    (common when there are rare classes with only 1 sample).
    """
    from neurosymbolic_iot.data_processing.splits import split_train_val_test

    try:
        return split_train_val_test(
            df,
            train=train,
            val=val,
            test=test,
            seed=seed,
            stratify_col=stratify_col,
        )
    except Exception as e:
        log.warning("Split with stratify=%s failed (%s). Falling back to non-stratified split.", stratify_col, e)
        return split_train_val_test(
            df,
            train=train,
            val=val,
            test=test,
            seed=seed,
            stratify_col=None,
        )


def train_and_evaluate(
    *,
    dataset: str,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    out_dir: str | Path,
    seed: int,
    label_col: str = "label",
    model_name: Optional[str] = None,
) -> ModelArtifacts:
    out_dir = Path(out_dir) / dataset
    ensure_dir(out_dir)

    all_df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    feature_cols = infer_feature_cols(all_df, label_col=label_col)

    X_train, y_train = make_xy(df_train, feature_cols, label_col=label_col)
    X_val, y_val = make_xy(df_val, feature_cols, label_col=label_col) if not df_val.empty else (None, None)
    X_test, y_test = make_xy(df_test, feature_cols, label_col=label_col) if not df_test.empty else (None, None)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    model = build_model(dataset=dataset, seed=seed, model_name=model_name)
    model.fit(X_train, y_train_enc)

    metrics: Dict[str, Any] = {
        "dataset": dataset,
        "seed": int(seed),
        "n_train": int(len(df_train)),
        "n_val": int(len(df_val)),
        "n_test": int(len(df_test)),
        "n_features": int(len(feature_cols)),
        "n_classes": int(len(le.classes_)),
        "classes": [str(c) for c in le.classes_],
        "model": model.__class__.__name__,
        "model_params": getattr(model, "get_params", lambda: {})(),
    }

    confusion_val_csv: Optional[str] = None
    confusion_test_csv: Optional[str] = None

    if X_val is not None and y_val is not None and len(y_val) > 0:
        y_val_pred_enc = model.predict(X_val)
        y_val_pred = le.inverse_transform(y_val_pred_enc)
        metrics["val"] = compute_metrics(y_val, y_val_pred)
        confusion_val_csv = str(out_dir / "confusion_val.csv")
        save_confusion_csv(y_val, y_val_pred, labels=le.classes_, out_csv=confusion_val_csv)

    if X_test is not None and y_test is not None and len(y_test) > 0:
        y_test_pred_enc = model.predict(X_test)
        y_test_pred = le.inverse_transform(y_test_pred_enc)
        metrics["test"] = compute_metrics(y_test, y_test_pred)
        confusion_test_csv = str(out_dir / "confusion_test.csv")
        save_confusion_csv(y_test, y_test_pred, labels=le.classes_, out_csv=confusion_test_csv)

    model_path = out_dir / "model.joblib"
    le_path = out_dir / "label_encoder.joblib"
    feat_path = out_dir / "feature_cols.json"
    metrics_path = out_dir / "metrics.json"

    joblib.dump(model, model_path)
    joblib.dump(le, le_path)
    save_json(feat_path, list(feature_cols))
    save_json(metrics_path, metrics)

    log.info("[%s] Saved artifacts under %s", dataset, out_dir.as_posix())

    return ModelArtifacts(
        dataset=dataset,
        out_dir=str(out_dir),
        model_path=str(model_path),
        label_encoder_path=str(le_path),
        feature_cols_path=str(feat_path),
        metrics_path=str(metrics_path),
        confusion_val_csv=confusion_val_csv,
        confusion_test_csv=confusion_test_csv,
    )


def load_artifacts(out_dir: str | Path, dataset: str) -> Tuple[BaseEstimator, LabelEncoder, List[str]]:
    out_dir = Path(out_dir) / dataset
    model = joblib.load(out_dir / "model.joblib")
    le: LabelEncoder = joblib.load(out_dir / "label_encoder.joblib")
    feature_cols: List[str] = json.loads((out_dir / "feature_cols.json").read_text(encoding="utf-8"))
    return model, le, feature_cols
