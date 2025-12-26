from pathlib import Path

import fire
import mlflow
from dataset import ensure_data
from hydra import compose, initialize_config_dir
from pytorch_lightning.loggers import MLFlowLogger
from ultralytics import YOLO
from utils import ensure_yolo_weights, export_to_onnx, get_git_commit_id


def main(config_path: str = "../configs", config_name: str = "config") -> None:
    config_dir = Path(__file__).resolve().parent / config_path
    config_dir = config_dir.resolve()

    print(f"Loading Hydra config from: {config_dir}")
    initialize_config_dir(config_dir=str(config_dir), version_base=None)
    cfg = compose(config_name=config_name)

    print("Ensuring dataset is available...")
    ensure_data(cfg)

    print("Ensuring model weights are available...")
    weights_path = ensure_yolo_weights(cfg)

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.mlflow_tracking_uri,
        tags={"git_commit": get_git_commit_id()},
    )

    params = {
        "batch_size": cfg.training.batch_size,
        "num_workers": cfg.training.num_workers,
        "model_type": cfg.model.type,
        "model_size": cfg.model.size,
        "learning_rate": cfg.training.lr,
        "max_epochs": cfg.training.epochs,
        "git_commit": get_git_commit_id(),
    }
    mlflow_logger.log_hyperparams(params)

    model = YOLO(weights_path)

    data_yaml_path = cfg.data_yaml

    model.train(
        data=str(data_yaml_path),
        epochs=cfg.training.epochs,
        batch=cfg.training.batch_size,
        lr0=cfg.training.lr,
        workers=cfg.training.num_workers,
    )

    out_onnx_dir = cfg.export.onnx_output_dir
    print(f"Exporting trained model to ONNX: {out_onnx_dir}")
    export_to_onnx(model, cfg)

    try:
        mlflow.log_artifact(out_onnx_dir)
    except Exception:
        print("Failed to log onnx to mlflow (file may be missing)")


if __name__ == "__main__":
    fire.Fire({"train": main})
