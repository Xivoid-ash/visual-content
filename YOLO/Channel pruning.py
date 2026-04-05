# ====================== YOLOv8n 最初基础训练 ======================
# 纯原版训练，不修改任何代码，最稳定、最安全
# =================================================================

from ultralytics import YOLO


#model = YOLO("yolov8n.pt")
#model = YOLO(r"D:\deeplearning\ultralytics-8.3.163\runs\detect\train2\weights\best.pt")
model = YOLO(r"yolov8n_pruned.pt")
if __name__ == "__main__":
    model.train(
        data="voc.yaml",         # 数据集：voc.yaml / coco.yaml / 你的自定义.yaml
        epochs=20,              # 训练轮数
        imgsz=320,               # 图片尺寸
        batch=32,
        cache="ram",
        lr0=0.001,  # 小学习率
        lrf=0.01,
        workers=1,
    )
