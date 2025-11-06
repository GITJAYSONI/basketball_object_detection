from utils.video_utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer


def main():
    print("Starting main function...")
      
    #Read video
    print("Reading video file...")
    video_frames = read_video("input_video/video_1.mp4")
    print(f"Read {len(video_frames)} frames from video")
 
    # Initialize Player Tracker
    print("Initializing Player Tracker...")
    player_tracker = PlayerTracker(model_path="models/player_detector.pt")
 
    # Run player tracking
    print("Running player tracking...")
    player_tracks = player_tracker.get_object_tracks(video_frames,
                                             read_from_stub=True,
                                             stub_path="stubs/player_tracks_stub.pkl" )
    print("Player tracking completed")
    
    # Initialize Ball Tracker
    print("Initializing Ball Tracker...")
    ball_tracker = BallTracker(model_path="models/ball_detector.pt")
 
    # Run ball tracking
    print("Running ball tracking...")
    ball_tracks = ball_tracker.get_object_tracks(video_frames,
                                              read_from_stub=True,
                                              stub_path="stubs/ball_tracks_stub.pkl" )
    print("Ball tracking completed")
                                             
    # Remove wrong detections
    print("Removing wrong ball detections...")
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
 
    # Interpolate ball positions
    print("Interpolating ball positions...")
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)
    print("Ball position interpolation completed")

    #Draw output
    print("Initializing drawers...")
    # Initialize Player Tracks Drawer
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
 
    # Draw object tracks
    print("Drawing player tracks...")
    output_video_frames = player_tracks_drawer.draw(video_frames,player_tracks)
    print("Drawing ball tracks...")
    output_video_frames = ball_tracks_drawer.draw(output_video_frames,ball_tracks)
 
    # Save video
    print("Saving output video...")
    save_video(output_video_frames, "output_videos/output_video.avi")
    print("Video saved successfully")
 
if __name__ == "__main__":
    main()