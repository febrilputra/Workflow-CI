"""Package the exact training output and smoke-test MLflow HTTP serving."""
import argparse
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=["bundle", "smoke", "deployment"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:5001")
    parser.add_argument("--image")
    parser.add_argument("--commit")
    args = parser.parse_args()
    winner = json.loads((args.output_dir / "winner.json").read_text(encoding="utf-8"))
    if args.action == "bundle":
        destination = args.bundle_dir / "artifacts" / winner["run_id"]
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("model", "reports", "winner.json", "selected_params.json", "candidate_results.json",
                     "training_provenance.json", "test_predictions.csv", "deployment.json", "smoke_test.json"):
            source = args.output_dir / name
            if source.is_dir():
                shutil.copytree(source, destination / name, dirs_exist_ok=True)
            elif source.is_file():
                shutil.copy2(source, destination / name)
        preprocessing = destination / "preprocessing"
        preprocessing.mkdir(exist_ok=True)
        for name in ("preprocessor.joblib", "feature_schema.json", "data_manifest.json"):
            shutil.copy2(args.data_dir / name, preprocessing / name)
        checksums = {}
        for file in destination.rglob("*"):
            if file.is_file() and file.name != "artifact_checksums.json":
                if file.stat().st_size >= 95 * 1024 * 1024:
                    raise ValueError(f"Artifact too large for normal Git: {file.name}. Configure Git LFS first.")
                checksums[file.relative_to(destination).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
        (destination / "artifact_checksums.json").write_text(json.dumps(checksums, indent=2) + "\n", encoding="utf-8")
        print(destination)
    elif args.action == "deployment":
        receipt = {"run_id": winner["run_id"], "model_sha256": winner["model_sha256"],
                   "image": args.image, "git_commit": args.commit,
                   "dockerhub_url": f"https://hub.docker.com/r/{args.image.split(':')[0]}",
                   "serving_port": 8080}
        (args.output_dir / "deployment.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    else:
        import pandas as pd
        import mlflow.sklearn
        X = pd.read_csv(args.data_dir / "X_val.csv").head(3).astype("float64")
        payload = {"dataframe_split": {"columns": list(X.columns), "data": X.values.tolist()}}
        deadline = time.monotonic() + 180
        while True:
            try:
                with urllib.request.urlopen(args.url + "/ping", timeout=3) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise RuntimeError("MLflow serving did not become healthy within 180 seconds.")
                time.sleep(2)
        request = urllib.request.Request(args.url + "/invocations", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        predicted = result.get("predictions")
        expected = mlflow.sklearn.load_model(str(args.output_dir / "model")).predict(X).tolist()
        if predicted != expected:
            raise AssertionError(f"Container predictions {predicted} differ from saved model predictions {expected}.")
        receipt = {"run_id": winner["run_id"], "request": payload, "response": result,
                   "expected_predictions": expected, "matches_saved_model": True}
        (args.output_dir / "smoke_test.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"run_id": winner["run_id"], "predictions": predicted, "matches_saved_model": True}))


if __name__ == "__main__":
    main()
