from ultralytics import YOLO
import os

print("Current directory:", os.getcwd())
print("Files:", os.listdir())

# Check that your file exists
video_path = r"C:\Users\jayso\OneDrive\Desktop\basketball\input_video\video_1.mp4"

if not os.path.exists(video_path):
    raise FileNotFoundError(f" The video file was not found: {video_path}")

# Load model
model = YOLO('yolov8x.pt')

# Run prediction
results = model.predict(video_path, save=True)

print(results)
print('=========================')
for box in results[0].boxes:
    print(box)
