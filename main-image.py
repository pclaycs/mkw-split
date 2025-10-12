import cv2
import numpy as np
from utils import find_obs_camera

# def detect_with_mask(frame, template, mask, threshold=0.7):
#     """Template matching with mask - ignores masked areas"""
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
#     for scale in np.linspace(0.8, 1.2, 5):
#         resized_template = cv2.resize(gray_template, None, fx=scale, fy=scale)
#         resized_mask = cv2.resize(mask, None, fx=scale, fy=scale)
        
#         if resized_template.shape[0] > gray_frame.shape[0] or resized_template.shape[1] > gray_frame.shape[1]:
#             continue
        
#         # Use mask parameter
#         result = cv2.matchTemplate(gray_frame, resized_template, cv2.TM_CCOEFF_NORMED, mask=resized_mask)
#         _, max_val, _, _ = cv2.minMaxLoc(result)
        
#         if max_val >= threshold:
#             return True, max_val
    
#     return False, 0.0

def detect_with_mask(frame, template, mask, threshold=0.7):
    """Template matching with mask - single scale only"""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Single scale matching - no loop
    result = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED, mask=mask)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    
    if max_val >= threshold:
        return True, max_val
    
    return False, 0.0

def load_template_with_mask(path):
    """Load a PNG with transparency and extract mask"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    
    if img.shape[2] == 4:  # Has alpha channel
        # Split into BGR and alpha
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        
        # Create mask: 255 where text is, 0 where transparent
        _, mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
        
        return bgr, mask
    else:
        # No alpha channel - create full mask (all white)
        mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
        return img, mask

# Load templates with masks
print("Loading templates...")
you_lose_template, you_lose_mask = load_template_with_mask('images/crop-you-lose-nobg.png')
you_win_template, you_win_mask = load_template_with_mask('images/crop-you-win-nobg.png')
new_record_template, new_record_mask = load_template_with_mask('images/crop-new-record-nobg.png')
finish_template, finish_mask = load_template_with_mask('images/crop-finish-nobg.png')

print("Templates loaded successfully!")

# Open camera
camera_index = find_obs_camera()
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("Camera opened. Press 'q' to quit")

# Define ROI for result text
roi_x1, roi_y1 = 428, 358
roi_x2, roi_y2 = 1493, 562

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Extract ROI for result detection
    result_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # Check for each result screen (all using masks now)
    lose_detected, lose_conf = detect_with_mask(result_roi, you_lose_template, you_lose_mask)
    win_detected, win_conf = detect_with_mask(result_roi, you_win_template, you_win_mask)
    record_detected, record_conf = detect_with_mask(result_roi, new_record_template, new_record_mask)
    finish_detected, finish_conf = detect_with_mask(result_roi, finish_template, finish_mask)
    
    # Print detections
    if lose_detected:
        print(f"❌ YOU LOSE detected! (confidence: {lose_conf:.2f})")
    
    if win_detected:
        print(f"🏆 YOU WIN detected! (confidence: {win_conf:.2f})")
    
    if record_detected:
        print(f"⭐ NEW RECORD detected! (confidence: {record_conf:.2f})")

    if finish_detected:
        print(f"🏁 FINISH detected! (confidence: {finish_conf:.2f})")
    
    
    # Draw rectangle on frame to visualize ROI
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
    
    # Display frame
    cv2.imshow('Game Feed', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()