import cv2
from utils import find_obs_camera

def select_roi_interactive():
    cap = cv2.VideoCapture(find_obs_camera(), cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # OpenCV's built-in ROI selector
        roi = cv2.selectROI("Select Region", frame, fromCenter=False)
        cv2.destroyAllWindows()
        
        x, y, w, h = roi
        print(f"ROI: x={x}, y={y}, w={w}, h={h}")
        print(f"Code: frame[{y}:{y+h}, {x}:{x+w}]")
        
        return roi
    
    return None

# Usage
roi = select_roi_interactive()