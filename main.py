from utils.video_utils import read_video, save_video
from trackers import PlayerTracker

def main():
      
    #Read video
    video_frames = read_video("input_video/video_1.mp4")

    # Initialize Player Tracker
    player_tracker = PlayerTracker(model_path="models/player_detector.pt")

    # Run player tracking
    player_tracker = player_tracker.get_object_tracks(video_frames) 

    print(player_tracker)
    
   # Save video
    save_video(video_frames, "output_videos/video1_output.avi")

if __name__ == "__main__":
    main()