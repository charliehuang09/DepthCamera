from ultralytics import YOLO

def download_model(path: str) -> None:
    model = YOLO(path)

    # Export the model to TensorRT
    model.export(format="onnx", nms=True)  # creates 'yolo11n.engine'

    # Load the exported TensorRT model
    # trt_model = YOLO("yolo11n.engine")

    # Run inference
    # results = trt_model("https://ultralytics.com/images/bus.jpg")

# download_model("yolo11n.pt")
# download_model("yolov8n.pt")
# download_model("game_peice.pt")
model = YOLO("game_peice.pt")
model.export(
    format="onnx",
    nms=True
)
