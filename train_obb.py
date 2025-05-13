from ultralytics import YOLO, settings

settings.update({"wandb": True})

# Create a new YOLO11n-OBB model from scratch
model = YOLO("yolov8n-obb.pt")

# Train the model on the DOTAv1 dataset
results = model.train(data="horizon-obb-medium.yaml", epochs=20, imgsz=1024, mosaic=1, multi_scale=True, degrees=25)
