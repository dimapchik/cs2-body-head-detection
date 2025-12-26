import fire
from ultralytics import YOLO


def export(model_weights: str = "yolov8n.pt", output_format: str = "onnx"):
    model = YOLO(model_weights)
    print("Exporting model", model_weights, "to format", output_format)
    try:
        model.export(format=output_format, imgsz=640)
        print("Export finished")
    except Exception as e:
        print("Export failed:", e)


if __name__ == "__main__":
    fire.Fire({"export": export})
