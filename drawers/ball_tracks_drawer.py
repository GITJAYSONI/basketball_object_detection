from .utils import draw_triangle # <-- FIX 1: Renamed 'draw_traingle' to 'draw_triangle'

class BallTracksDrawer:
    """
    A drawer class responsible for drawing ball tracks on video frames.
    """

    def __init__(self):
        self.ball_pointer_color = (0, 255, 0) # RGB Green color for ball pointer

    def draw(self, video_frames, tracks):
        """
        Draws ball pointers on each video frame based on provided tracking information.
        """

        output_video_frames = []
        
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            ball_dict = tracks[frame_num]

            # Draw ball 
            for _, ball in ball_dict.items():
                if ball["bbox"] is None:
                    continue
                frame = draw_triangle(frame, ball["bbox"], self.ball_pointer_color) # <-- FIX 2: Renamed 'draw_traingle' to 'draw_triangle'

            output_video_frames.append(frame)
            
        return output_video_frames