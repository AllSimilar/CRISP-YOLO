### YOLO 训练
from ultralytics import YOLO
from ultralytics import RTDETR
# 加载自己的架构
model = YOLO("")  # load a pretrained model (recommended for training)

model.load("yolo26s.pt")

# 直接加载官方权重
# model = YOLO("yolo26m.pt")
# model = RTDETR("rtdetr-l.pt")
# Train the model
# results = model.train(data="HazyDet.yaml", epochs=150, imgsz=768)
# results = model.train(data="FFA_Net.yaml", epochs=150, imgsz=768)
results = model.train(data="RDDTS.yaml", epochs=100, imgsz=768)
# results = model.train(data="GridDehaze.yaml", 
#                     epochs=100, 
#                     imgsz=768, deterministic=False)
