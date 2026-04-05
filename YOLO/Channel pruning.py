from ultralytics import YOLO


#model = YOLO("yolov8n.pt")
#model = YOLO(r"D:\deeplearning\ultralytics-8.3.163\runs\detect\train2\weights\best.pt")
model = YOLO(r"yolov8n_pruned.pt")
if __name__ == "__main__":
    model.train(
        data="voc.yaml",         # 数据集
        epochs=20,              # 训练轮数
        imgsz=320,               # 图片尺寸
        batch=32,
        cache="ram",
        lr0=0.001,  # 小学习率
        lrf=0.01,
        workers=1,
    )
