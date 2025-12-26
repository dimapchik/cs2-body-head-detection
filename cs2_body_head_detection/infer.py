from pathlib import Path

import fire
from hydra import compose, initialize_config_dir
from ultralytics import YOLO
from utils import ensure_data


def infer(source: str = "data", save_dir: str = "runs/detect") -> None:
    config_path: str = "../configs"
    config_name: str = "config"
    config_dir = Path(__file__).resolve().parent / config_path
    config_dir = config_dir.resolve()
    print(f"Loading Hydra config from: {config_dir}")
    initialize_config_dir(config_dir=str(config_dir), version_base=None)
    cfg = compose(config_name=config_name)

    ensure_data()

    model = YOLO(cfg.export.onnx_path)
    _ = model(source, save=True, project=save_dir)


if __name__ == "__main__":
    fire.Fire(infer)
