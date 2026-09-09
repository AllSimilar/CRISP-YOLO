from ultralytics import YOLO

# test
# Load a model
model = YOLO("")  # load a pretrained model (recommended for training)

# Train the model
results = model.val(data="HazyDet.yaml", imgsz=768)
# results = model.val(data="RDDTS.yaml", imgsz=768)
# results = model.val(data="FFA_Net.yaml", imgsz=768)
# results = model.val(data="GridDehaze.yaml", imgsz=768)
