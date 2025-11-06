import cv2
import numpy as np
import sys
sys.path.append("../")          #"../" to the system path so you can import utilities from a parent directory.
from utils import get_center_of_bbox, get_bbox_width

def draw_triangle(frame, bbox, color, angle=90, offset=6):
    """
    Draw a triangle pointer near the bbox.
    - angle: direction in degrees (90 = up, 0 = right, 180 = left, 270 = down).
    - offset: pixels of separation between the triangle base and the bbox edge.
    By default the triangle points up (90°) and is placed just above the ball to avoid overlap.
    """
    x_center, y_center = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)

    # base size relative to bbox width
    size = max(8, int(width * 0.18))

    # approximate radius/half-height of the bbox to compute spacing
    radius = int(width / 2)

    # support only the common angles (up/right/down/left) for simplicity
    a = angle % 360 
    if a == 90:  # up 
        # place base just above the top of the bbox
        ball_top = int(y_center - radius)
        base_y = ball_top - offset
        apex_y = base_y - (2 * size)         
        apex = (int(x_center), int(apex_y))  
        base_left = (int(x_center - size), int(base_y))
        base_right = (int(x_center + size), int(base_y))
        triangle_points = np.array([apex, base_left, base_right])
    elif a == 0:  # right
        apex = (int(x_center + radius + offset + size), int(y_center))
        base_top = (int(x_center + radius + offset - size), int(y_center - size))
        base_bottom = (int(x_center + radius + offset - size), int(y_center + size))
        triangle_points = np.array([apex, base_top, base_bottom])
    elif a == 180:  # left
        apex = (int(x_center - radius - offset - size), int(y_center))
        base_top = (int(x_center - radius - offset + size), int(y_center - size))
        base_bottom = (int(x_center - radius - offset + size), int(y_center + size))
        triangle_points = np.array([apex, base_top, base_bottom])
    elif a == 270:  # down
        ball_bottom = int(y_center + radius)
        base_y = ball_bottom + offset
        apex_y = base_y + (2 * size)
        apex = (int(x_center), int(apex_y))
        base_left = (int(x_center - size), int(base_y))
        base_right = (int(x_center + size), int(base_y))
        triangle_points = np.array([apex, base_right, base_left])
    else:
        # fallback: draw an upward triangle
        ball_top = int(y_center - radius)
        base_y = ball_top - offset
        apex_y = base_y - (2 * size)
        apex = (int(x_center), int(apex_y))
        base_left = (int(x_center - size), int(base_y))
        base_right = (int(x_center + size), int(base_y))
        triangle_points = np.array([apex, base_left, base_right])

    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

    return frame

def draw_ellipse(frame, bbox, color, track_id=None):  
    y2 = int(bbox[3])
    x_center,_ = get_center_of_bbox(bbox)
    width =get_bbox_width(bbox)

 # code to drw ellipse under the player
    cv2.ellipse(
        frame,
        center=(x_center,y2),
        axes=(int(width), int(0.35*width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color = color,
        thickness=2,
        lineType=cv2.LINE_4
    )

    rectangle_width = 40
    rectangle_height=20
    x1_rect = x_center - rectangle_width//2
    x2_rect = x_center + rectangle_width//2
    y1_rect = (y2- rectangle_height//2) +15
    y2_rect = (y2+ rectangle_height//2) +15

    if track_id is not None:
        cv2.rectangle(frame,
                        (int(x1_rect),int(y1_rect) ),
                        (int(x2_rect),int(y2_rect)),
                        color,
                        cv2.FILLED)
        
        x1_text = x1_rect+12
        if track_id > 99:
            x1_text -=10
        
        # putText expects: img, text(str), org(tuple), font, fontScale, color, thickness, lineType
        cv2.putText(
            frame,
            str(track_id),
            (int(x1_text), int(y1_rect+15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

    return frame