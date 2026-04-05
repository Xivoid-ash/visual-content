from ultralytics import YOLO

model = YOLO(r"D:\deeplearning\ultralytics-8.3.163\runs\detect\train11\weights\best.pt")
model.predict(
    source=r"D:\deeplearning\ultralytics-8.3.163\datasets\nailong\images\all",
    save=True,
    show=False,
    save_txt=True,
    )