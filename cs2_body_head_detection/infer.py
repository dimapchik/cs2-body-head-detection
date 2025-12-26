from pathlib import Path

import fire
from dataset import ensure_data
from hydra import compose, initialize_config_dir
from ultralytics import YOLO


def infer(source: str = "../data/val/images/screenshot_170.png") -> None:
    config_path: str = "../configs"
    config_name: str = "config"
    config_dir = Path(__file__).resolve().parent / config_path
    config_dir = config_dir.resolve()
    print(f"Loading Hydra config from: {config_dir}")
    initialize_config_dir(config_dir=str(config_dir), version_base=None)
    cfg = compose(config_name=config_name)

    print("Ensuring dataset is available...")
    ensure_data(cfg)

    print("Loading model for inference...")
    model = YOLO(cfg.inference.weights_path)
    _ = model.predict(
        source, save=True, save_dir=cfg.inference.save_dir, name="results"
    )


if __name__ == "__main__":
    fire.Fire(infer)
