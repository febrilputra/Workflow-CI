"""Shared data contract and explicit MLflow evaluation for Telco Churn."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.inspection import permutation_importance
from sklearn.utils import estimator_html_repr


MODEL_REQUIREMENTS = [
    "mlflow==2.19.0", "scikit-learn==1.5.2", "numpy==1.26.4",
    "pandas==2.2.3", "scipy==1.14.1", "cloudpickle==3.1.0", "joblib==1.4.2",
]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def parser(description: str, script_dir: Path) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--data-dir", type=Path, default=script_dir / "telco_churn_preprocessing")
    result.add_argument("--output-dir", type=Path, default=script_dir / "output")
    result.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    result.add_argument("--experiment", default=os.getenv("MLFLOW_EXPERIMENT_NAME", "telco-churn"))
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--n-jobs", type=int, default=2)
    return result


def configure_tracking(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tracking_uri = args.tracking_uri or (output.parent / "mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    if not os.getenv("MLFLOW_RUN_ID"):
        mlflow.set_experiment(args.experiment)
    return output


def load_data(data_dir: Path) -> tuple[dict, dict]:
    data_dir = data_dir.resolve()
    names = [f"{kind}_{split}.csv" for split in ("train", "val", "test") for kind in ("X", "y")]
    names += ["feature_schema.json", "data_manifest.json", "preprocessor.joblib"]
    missing = [name for name in names if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete preprocessed dataset: {missing}. Run preprocessing first.")
    manifest = json.loads((data_dir / "data_manifest.json").read_text(encoding="utf-8"))
    hashes = {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in names}
    for name, expected in manifest.get("files", {}).items():
        if name in hashes and isinstance(expected, str) and hashes[name] != expected:
            raise ValueError(f"Checksum mismatch for {name}; copy the complete original preprocessing output.")
    schema = json.loads((data_dir / "feature_schema.json").read_text(encoding="utf-8"))
    data = {}
    for split in ("train", "val", "test"):
        X = pd.read_csv(data_dir / f"X_{split}.csv")
        y_frame = pd.read_csv(data_dir / f"y_{split}.csv")
        if list(y_frame.columns) != ["Churn"]:
            raise ValueError(f"y_{split}.csv must have exactly the Churn column.")
        y = y_frame["Churn"]
        if len(X) != len(y) or not len(X) or set(y.unique()) != {0, 1}:
            raise ValueError(f"Invalid {split} row count or binary labels.")
        if X.columns.duplicated().any() or {"Churn", "customerID"}.intersection(X.columns):
            raise ValueError("Target/customer identifiers cannot be model features.")
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
            raise ValueError(f"{split} has nonnumeric columns; model expects preprocessed features.")
        X = X.astype("float64")
        if not np.isfinite(X.to_numpy()).all():
            raise ValueError(f"{split} contains missing or infinite features.")
        expected_features = schema.get("encoded_features")
        if expected_features and list(X.columns) != expected_features:
            raise ValueError(f"{split} feature order differs from feature_schema.json.")
        if data and list(X.columns) != list(data["train"][0].columns):
            raise ValueError("All splits must have identical feature order.")
        data[split] = (X, y.astype("int64"))
    provenance = {"files": hashes, "rows": {key: len(value[0]) for key, value in data.items()},
                  "feature_count": data["train"][0].shape[1], "source_manifest": manifest,
                  "python": platform.python_version(), "sklearn": sklearn.__version__}
    source_dir = Path(__file__).resolve().parent
    source_files = list(source_dir.glob("*.py")) + [source_dir / "requirements.txt"]
    provenance["source_files_sha256"] = {
        file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in source_files if file.is_file()
    }
    return data, provenance


def evaluate(model, X: pd.DataFrame, y: pd.Series, prefix: str) -> tuple[dict, np.ndarray, np.ndarray]:
    predicted = model.predict(X)
    probability = model.predict_proba(X)[:, list(model.classes_).index(1)]
    values = {
        "score": metrics.accuracy_score(y, predicted),
        "accuracy_score": metrics.accuracy_score(y, predicted),
        "precision_score": metrics.precision_score(y, predicted, average="weighted", zero_division=0),
        "recall_score": metrics.recall_score(y, predicted, average="weighted", zero_division=0),
        "f1_score": metrics.f1_score(y, predicted, average="weighted", zero_division=0),
        "log_loss": metrics.log_loss(y, probability, labels=[0, 1]),
        "roc_auc": metrics.roc_auc_score(y, probability),
        "average_precision": metrics.average_precision_score(y, probability),
        "churn_precision": metrics.precision_score(y, predicted, pos_label=1, zero_division=0),
        "churn_recall": metrics.recall_score(y, predicted, pos_label=1, zero_division=0),
        "churn_f1": metrics.f1_score(y, predicted, pos_label=1, zero_division=0),
    }
    return {f"{prefix}_{key}": float(value) for key, value in values.items()}, predicted, probability


def log_provenance(data_dir: Path, data: dict, provenance: dict, output: Path) -> None:
    write_json(output / "training_provenance.json", provenance)
    mlflow.log_artifact(str(output / "training_provenance.json"))
    for name in provenance.get("source_files_sha256", {}):
        mlflow.log_artifact(str(Path(__file__).resolve().parent / name), artifact_path="source_snapshot")
    for name in ("feature_schema.json", "data_manifest.json", "preprocessor.joblib"):
        mlflow.log_artifact(str(data_dir / name), artifact_path="preprocessing")
    train_frame = data["train"][0].assign(Churn=data["train"][1].to_numpy())
    dataset = mlflow.data.from_pandas(train_frame, name="telco_churn_train", targets="Churn")
    mlflow.log_input(dataset, context="training")
    mlflow.set_tags({"dataset": "IBM Telco Customer Churn", "student": "Muhammad-Febrilian-Kurnia-Putra",
                     "dicoding_username": "febril_putra", "test_used_for_selection": "false",
                     "raw_sha256": provenance["source_manifest"].get("raw_sha256", "see data_manifest"),
                     "estimator_name": "RandomForestClassifier",
                     "estimator_class": "sklearn.ensemble._forest.RandomForestClassifier"})


def manual_artifacts(model, data: dict, output: Path, evaluations: dict, seed: int, n_jobs: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "estimator.html").write_text(estimator_html_repr(model), encoding="utf-8")
    descriptions = {}
    for key in evaluations:
        descriptions[key] = {
            "split": key.split("_", 1)[0],
            "averaging": "weighted for *_precision_score/*_recall_score/*_f1_score; binary positive=1 for churn_*",
            "definition": "training_common.evaluate; threshold=0.5; average_precision is sklearn AP (not trapezoidal PR AUC)",
        }
    write_json(output / "metric_info.json", descriptions)
    X_train, y_train = data["train"]
    training_predicted = model.predict(X_train)
    training_probability = model.predict_proba(X_train)[:, 1]
    plotters = {
        "training_confusion_matrix.png": lambda ax: metrics.ConfusionMatrixDisplay.from_predictions(y_train, training_predicted, display_labels=["No", "Yes"], ax=ax),
        "training_roc_curve.png": lambda ax: metrics.RocCurveDisplay.from_predictions(y_train, training_probability, ax=ax),
        "training_precision_recall_curve.png": lambda ax: metrics.PrecisionRecallDisplay.from_predictions(y_train, training_probability, ax=ax),
    }
    for name, render in plotters.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        render(ax)
        fig.tight_layout()
        fig.savefig(output / name, dpi=150)
        plt.close(fig)
    X_val, y_val = data["val"]
    reports = {"validation": metrics.classification_report(y_val, model.predict(X_val),
               target_names=["No", "Yes"], output_dict=True, zero_division=0)}
    write_json(output / "classification_report.json", reports)
    # Explanations use validation only; the test set remains exclusively final evaluation.
    importance = permutation_importance(model, X_val, y_val, scoring="average_precision",
                                        n_repeats=3, max_samples=min(500, len(X_val)),
                                        random_state=seed, n_jobs=n_jobs)
    ranks = np.argsort(importance.importances_mean)[-15:]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(np.asarray(X_val.columns)[ranks], importance.importances_mean[ranks],
            xerr=importance.importances_std[ranks])
    ax.set(xlabel="Decrease in validation average precision", title="Permutation importance (validation, 3 repeats)")
    fig.tight_layout()
    fig.savefig(output / "permutation_importance.png", dpi=150)
    plt.close(fig)
    write_json(output / "permutation_importance.json", {
        "data": "validation only", "scoring": "average_precision", "n_repeats": 3,
        "note": "Correlated/one-hot features may share importance; this is predictive association, not causality.",
        "features": [{"name": name, "mean": float(mean), "std": float(std)} for name, mean, std in
                     zip(X_val.columns, importance.importances_mean, importance.importances_std)],
    })
    for file in output.iterdir():
        if file.is_file():
            mlflow.log_artifact(str(file))


def export_model(run_id: str, output: Path, metadata: dict) -> None:
    # Download this exact run, never select an arbitrary latest run from the experiment.
    local_artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model", dst_path=str(output))
    metadata.update({"run_id": run_id, "model_uri": f"runs:/{run_id}/model",
                     "artifact_uri": mlflow.get_artifact_uri("model"), "local_model_path": "model",
                     "tracking_uri": mlflow.get_tracking_uri(),
                     "model_sha256": hashlib.sha256((Path(local_artifact_path) / "model.pkl").read_bytes()).hexdigest()})
    write_json(output / "winner.json", metadata)
    mlflow.log_artifact(str(output / "winner.json"))
    print(json.dumps({"run_id": run_id, "model_uri": metadata["model_uri"],
                      "output_dir": str(output)}, indent=2))
