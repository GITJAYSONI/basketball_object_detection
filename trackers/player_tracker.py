from ultralytics import YOLO
import supervision as sv       #a library with tools for tracking, drawing boxes, etc.
import sys                     
sys.path.append("../")          #"../" to the system path so you can import utilities from a parent directory.
from utils import read_stub, save_stub   #read_stub / save_stub: helper functions for loading or saving cached data (probably pickle or JSON files)

class PlayerTracker:
  def __init__(self, model_path):
    self.model = YOLO(model_path)
    self.tracker = sv.ByteTrack()     #tracker (a multi-object tracking algorithm that links detections across frames).{best.pt}
    
  def detect_frames(self, frame):
    batch_size = 20
    detections = []
    for i in range(0,len(frame),batch_size):
      batch_frames = frame[i:i+batch_size]           # it was slicing the frame If i=0 and batch_size=20, it takes frames[0:20].
      batch_detections = self.model.predict(batch_frames, conf=0.5)
      detections+=batch_detections
    return detections
  
  def get_object_tracks(self, frames,read_from_stub = False, stub_path = None):           #Before doing expensive detection and tracking, the function first checks if cached results exist and are valid.

    tracks = read_stub(read_from_stub, stub_path)
    if tracks is not None and isinstance(tracks, list):
      if len(tracks) == len(frames):                                   #This checks if stub data was successfully loaded.
                              #Ensures the loaded stub data matches the number of frames you’re currently trying to process.
           #If there’s already saved (cached) tracking data available from a previous run,
          #and it matches the number of frames we’re processing now,
          #then skip running YOLO and ByteTrack again — just use the old results.
          return tracks

    detections =self.detect_frames(frames)   
    tracks = []

    for frame_num, detection in enumerate(detections):
      cls_names= detection.names
      cls_names_inv ={v:k for k,v in cls_names.items()}

      detection_supervision = sv.Detections.from_ultralytics(detection)

      detection_with_tracks= self.tracker.update_with_detections(detection_supervision)    #his part of the function runs YOLO on all video frames to detect players and then uses ByteTrack to assign a unique tracking ID to each detected object, so they can be followed across frames.

      tracks.append({})
      for frame_detection in detection_with_tracks:
          bbox = frame_detection[0].tolist()
          cls_id = frame_detection[3]
          track_id = frame_detection[4] 

          if cls_id == cls_names_inv['Player']:
            tracks[frame_num][track_id] = {"box" : bbox}


    save_stub(stub_path, tracks)     
    return tracks               #finds the players and follows them through the video
  


  