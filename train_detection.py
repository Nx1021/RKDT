from __init__ import CFG_DIR, SCRIPT_DIR
from ultralytics import YOLO
if __name__ == '__main__':
    # Load a model
    # model = YOLO("yolov8l.yaml")  # build a new model from scratch
    model = YOLO(f"{SCRIPT_DIR}/weights/yolov8l.pt")  # load a pretrained model (recommended for training)

    model.train(cfg = f"{CFG_DIR}/yolo_linemod.yaml")  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    path = model.export(format="onnx")  # export the model to ONNX format
