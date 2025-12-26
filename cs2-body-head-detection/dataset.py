import shutil
import subprocess
from pathlib import Path

import kagglehub
from git import Repo
from omegaconf import DictConfig


def ensure_data(cfg: DictConfig) -> None:
    data_path = Path(cfg.data.path)
    if data_path.exists() and any(data_path.iterdir()):
        print(f"Data already exists at {data_path}, skipping download")
        return

    print("Data not found locally. Attempting to pull via DVC...")
    pulled = False
    try:
        if Repo is not None:
            repo = Repo(".")
            repo.pull()
            pulled = True
    except Exception as e:
        print("DVC Python API pull failed:", e)

    if not pulled:
        ret = subprocess.run(["dvc", "pull"], cwd=Path.cwd())
        if ret.returncode == 0:
            pulled = True
        else:
            print("dvc pull failed (exit code", ret.returncode, ")")

    if not pulled:
        print("Falling back to Kaggle download...")
        download_data(cfg)


def download_data(cfg: DictConfig) -> None:
    if kagglehub is None:
        print("kagglehub not installed.")
        return

    target = Path(cfg.data.path)

    if target.exists() and any(target.iterdir()):
        print(f"Data already exists at {target}, skipping download")
        return

    try:
        print(f"Downloading dataset from Kaggle: {cfg.dataset_ref}")
        local_path = kagglehub.dataset_download(cfg.dataset_ref)
        print(f"Dataset downloaded to: {local_path}")

        target.mkdir(parents=True, exist_ok=True)

        for item in Path(local_path).iterdir():
            if item.is_dir():
                dst = target / item.name
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, target)

        print(f"Dataset successfully copied to {target}")
    except Exception as e:
        print(f"Failed to download dataset from Kaggle: {e}")
        print("Make sure you have kagglehub installed: pip install kagglehub")
        print("And that you're authenticated with Kaggle API")
