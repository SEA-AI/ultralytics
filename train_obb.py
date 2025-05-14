from ultralytics import YOLO, settings
import wandb
import os

wandb.login(key=os.environ.get('WANDB_API_KEY'))

settings.update({"wandb": True})

# Create a new YOLO11n-OBB model from scratch
model = YOLO("yolov8n-obb.pt")

# Train the model on the DOTAv1 dataset
results = model.train(data="horizon-obb-large.yaml", epochs=25, imgsz=960, mosaic=1, multi_scale=True, degrees=25, fill_value=0)
