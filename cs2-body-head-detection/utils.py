import shutil
import subprocess
from pathlib import Path

import mlflow
from omegaconf import DictConfig
from ultralytics import YOLO


def setup_mlflow(
    tracking_uri: str = "MLFLOW_TRACKING_URI", experiment_name: str = "default"
):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def export_to_onnx(model, output_path: str, imgsz: int = 640):
    try:
        model.export(format="onnx", imgsz=imgsz)
        possible = list(Path(".").rglob("*.onnx"))
        if possible:
            src = possible[-1]
            dst = Path(output_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.replace(dst)
            print(f"ONNX exported to {dst}")
        else:
            print("Export completed but no .onnx file found automatically;")
    except Exception as e:
        print("export_to_onnx failed:", e)


def get_git_commit_id() -> str:
    arr = ["git", "rev-parse", "HEAD"]
    try:
        rev = subprocess.check_output(arr).decode().strip()
        return rev
    except Exception:
        return "unknown"


def ensure_yolo_weights(cfg: DictConfig) -> str:
    model_size = cfg.model.size
    weights_dir = cfg.model.weights_dir
    weights_path = Path(weights_dir) / f"yolov8{model_size}.pt"

    if weights_path.exists():
        print(f"Using cached weights: {weights_path}")
        return str(weights_path)

    print(f"Weights not found. Downloading yolov8{model_size}")
    try:
        Path(weights_dir).mkdir(parents=True, exist_ok=True)

        _ = YOLO(f"{weights_path}/yolov8{model_size}.pt")

        ultralytics_cache = Path.home() / ".cache" / "ultralytics" / "weights"

        if ultralytics_cache.exists():
            src_weights = ultralytics_cache / f"yolov8{model_size}.pt"
            if src_weights.exists():
                shutil.copy2(src_weights, weights_path)
                print(f"Weights cached at {weights_path}")
                return str(weights_path)

        print(f"yolov8{model_size}.pt loaded; using from Ultralytics cache")
        return f"yolov8{model_size}.pt"

    except Exception as e:
        print(f"Failed to download YOLOv8 weights: {e}")
        print(f"Falling back to model name: yolov8{model_size}.pt")
        return f"yolov8{model_size}.pt"
