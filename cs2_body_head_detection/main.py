import fire
from infer import infer
from train import main as train

if __name__ == "__main__":
    fire.Fire({"train": train, "infer": infer})
