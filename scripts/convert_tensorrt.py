import subprocess

import fire


def convert(onnx_path: str, trt_path: str, fp16: bool = False):
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={trt_path}",
        "--workspace=2048",
    ]
    if fp16:
        cmd.append("--fp16")
    try:
        subprocess.check_call(cmd)
        print("TensorRT engine saved to", trt_path)
    except FileNotFoundError:
        print("trtexec not found.")
    except subprocess.CalledProcessError as e:
        print("trtexec failed:", e)


if __name__ == "__main__":
    fire.Fire({"convert": convert})
