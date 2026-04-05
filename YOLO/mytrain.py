
from ultralytics import YOLO

if __name__ =="__main__":
    model = YOLO("YOLOv8n.pt")
    model.train(
        data=r"nailong_1.yaml",
        epochs=30,
        imgsz=640,
        batch=-1,
        cache="ram",
        workers=1,
    )