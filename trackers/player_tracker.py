from ultralytics import YOLO
import supervision as sv       #a library with tools for tracking, drawing boxes, etc.
import sys                     
sys.path.append("../")          #"../" to the system path so you can import utilities from a parent directory.
from utils import read_stub, save_stub   #read_stub / save_stub: helper functions for loading or saving cached data (probably pickle or JSON files)

class PlayerTracker:
  def __init__(self, model_path):
    self.model = YOLO(model_path)
    self.tracker = sv.ByteTrack()     #tracker (a multi-object tracking algorithm that links detections across frames).{best.pt}
 
  def detect_frames(self, frames):
      batch_size=20 
      detections = [] 
      for i in range(0,len(frames),batch_size):
          detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.5)
          detections += detections_batch
      return detections

  def get_object_tracks(self, frames,read_from_stub = False, stub_path = None):           #Before doing expensive detection and tracking, the function first checks if cached results exist and are valid.

    tracks = read_stub(read_from_stub, stub_path)
    if tracks is not None and len(tracks) == len(frames):
        return tracks

    detections = self.detect_frames(frames)

    tracks=[]

    for frame_num, detection in enumerate(detections):
        cls_names = detection.names
        cls_names_inv = {v:k for k,v in cls_names.items()}

        # Covert to supervision Detection format
        detection_supervision = sv.Detections.from_ultralytics(detection)

        # Track Objects
        detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

        tracks.append({})

        for frame_detection in detection_with_tracks:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
            track_id = frame_detection[4]

            if cls_id == cls_names_inv['Player']:
                tracks[frame_num][track_id] = {"bbox":bbox}
    
    if stub_path is not None:
        save_stub(stub_path,tracks)
    return tracks