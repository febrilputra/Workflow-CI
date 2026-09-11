"""Bounded manual tuning; validation selects winner and test is evaluated once."""
from pathlib import Path
import json

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid

from training_common import (MODEL_REQUIREMENTS, configure_tracking, evaluate, export_model,
    load_data, log_provenance, log_source_versioning, manual_artifacts, parser, write_json)


def main() -> None:
    argument_parser = parser(__doc__, Path(__file__).resolve().parent)
    argument_parser.add_argument("--params-file", type=Path,
        help="Retrain one previously selected parameter set (used by CI); skip hyperparameter search.")
    args = argument_parser.parse_args()
    output = configure_tracking(args)
    data, provenance = load_data(args.data_dir)
    mlflow.autolog(disable=True)
    mlflow.sklearn.autolog(disable=True)
    if args.params_file:
        parameters = json.loads(args.params_file.read_text(encoding="utf-8"))
        candidates = [parameters]
    else:
        candidates = list(ParameterGrid({"n_estimators": [200], "max_depth": [8, 16, None],
                         "min_samples_leaf": [1, 4], "class_weight": [None, "balanced"]}))
    selected = None
    table = []
    stage = "ci_retraining" if args.params_file else "manual_tuning"
    with mlflow.start_run(run_name=stage) as parent:
        mlflow.set_tags({"stage": stage, "logging": "manual", "selection_metric": "validation_average_precision"})
        mlflow.log_params({"candidate_count": len(candidates), "selection_metric": "validation_average_precision",
                           "selection_split": "validation", "test_selection": False})
        log_provenance(args.data_dir, data, provenance, output)
        for index, candidate in enumerate(candidates):
            params = {**candidate, "random_state": args.seed, "n_jobs": args.n_jobs}
            with mlflow.start_run(run_name=f"candidate-{index:02d}", nested=True) as child:
                log_source_versioning()
                model = RandomForestClassifier(**params)
                mlflow.log_params(model.get_params(deep=True))
                model.fit(*data["train"])
                training_metrics, _, _ = evaluate(model, *data["train"], "training")
                validation_metrics, _, _ = evaluate(model, *data["val"], "validation")
                mlflow.log_metrics(training_metrics | validation_metrics)
                row = {"candidate": index, "run_id": child.info.run_id, "parameters": candidate,
                       **validation_metrics}
                table.append(row)
                # Deterministic first candidate wins exact ties; test never participates.
                if selected is None or validation_metrics["validation_average_precision"] > selected["score"]:
                    selected = {"model": model, "score": validation_metrics["validation_average_precision"],
                                "candidate": row, "metrics": training_metrics | validation_metrics}
        model = selected["model"]
        mlflow.log_params(model.get_params(deep=True))
        test_metrics, test_prediction, _ = evaluate(model, *data["test"], "test")
        all_metrics = selected["metrics"] | test_metrics
        mlflow.log_metrics(all_metrics)
        write_json(output / "candidate_results.json", table)
        write_json(output / "selected_params.json", model.get_params(deep=True))
        mlflow.log_artifact(str(output / "candidate_results.json"))
        mlflow.log_artifact(str(output / "selected_params.json"))
        manual_artifacts(model, data, output / "reports", all_metrics, args.seed, args.n_jobs)
        # Test predictions are audit output after selection, never another tuning input.
        pd.DataFrame({"actual": data["test"][1], "predicted": test_prediction}).to_csv(
            output / "test_predictions.csv", index=False)
        mlflow.log_artifact(str(output / "test_predictions.csv"))
        example = data["train"][0].head(3)
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=example,
            signature=infer_signature(example, model.predict(example)), pip_requirements=MODEL_REQUIREMENTS)
        export_model(parent.info.run_id, output, {
            "stage": stage, "selected_params": model.get_params(deep=True), "metrics": all_metrics,
            "selected_candidate_run_id": selected["candidate"]["run_id"],
            "selection_metric": "validation_average_precision", "test_evaluated": True,
            "training_data": "train only; validation used for selection; final test evaluated once",
        })


if __name__ == "__main__":
    main()
