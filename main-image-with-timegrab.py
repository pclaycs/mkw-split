import cv2
import numpy as np
from utils import find_obs_camera
import time

last_detection_time = {
    'lose': 0,
    'win': 0,
    'record': 0,
    'finish': 0
}
COOLDOWN_SECONDS = 5

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

def load_digit_templates():
    """Load digit templates 0-9 only"""
    templates = {}
    for i in range(10):
        img = cv2.imread(f'images/timestamps/cropped/{i}.png', cv2.IMREAD_GRAYSCALE)
        if img is not None:
            templates[str(i)] = img
    
    return templates

def recognize_character(roi, templates):
    """Match a character ROI against all templates"""
    # ROI is already thresholded grayscale from extract_timestamp
    
    best_match = None
    best_score = 0
    
    for char, template in templates.items():
        # Resize TEMPLATE to match ROI size (better than resizing ROI)
        resized_template = cv2.resize(template, (roi.shape[1], roi.shape[0]))
        
        # Compare directly (both are thresholded binary images now)
        result = cv2.matchTemplate(roi, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > best_score:
            best_score = max_val
            best_match = char
    
    return best_match, best_score

def extract_timestamp(frame, templates):
    """Extract timestamp from frame"""
    # Timestamp ROI
    timestamp_roi = frame[54:107, 1550:1864]
    
    # Preprocess
    gray = cv2.cvtColor(timestamp_roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours to locate each character
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours left to right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    digits = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Skip small contours (colon and period)
        if w < 15 or h < 20:
            continue
        
        # Extract character ROI
        char_roi = thresh[y:y+h, x:x+w]
        
        # Recognize character (digits only now)
        char, conf = recognize_character(char_roi, templates)
        
        if char and conf > 0.3:
            digits.append(char)
    
    # Reconstruct timestamp: M:SS.mmm
    if len(digits) == 6:
        timestamp = f"{digits[0]}:{digits[1]}{digits[2]}.{digits[3]}{digits[4]}{digits[5]}"
    else:
        timestamp = ''.join(digits)  # Fallback if wrong number of digits
    
    return timestamp

# Load templates with masks
print("Loading templates...")
you_lose_template, you_lose_mask = load_template_with_mask('images/crop-you-lose-nobg.png')
you_win_template, you_win_mask = load_template_with_mask('images/crop-you-win-nobg.png')
new_record_template, new_record_mask = load_template_with_mask('images/crop-new-record-nobg.png')
finish_template, finish_mask = load_template_with_mask('images/crop-finish-nobg.png')

# Load digit templates
digit_templates = load_digit_templates()

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

    current_time = time.time()
    
    # Check for each result screen (all using masks now)
    lose_detected, lose_conf = detect_with_mask(result_roi, you_lose_template, you_lose_mask)
    win_detected, win_conf = detect_with_mask(result_roi, you_win_template, you_win_mask)
    record_detected, record_conf = detect_with_mask(result_roi, new_record_template, new_record_mask)
    finish_detected, finish_conf = detect_with_mask(result_roi, finish_template, finish_mask)
    
    # Handle detections with timestamp extraction
    if lose_detected and (current_time - last_detection_time['lose']) > COOLDOWN_SECONDS:
        timestamp = extract_timestamp(frame, digit_templates)
        print(f"❌ YOU LOSE detected! Time: {timestamp} (confidence: {lose_conf:.2f})")
        last_detection_time['lose'] = current_time
    
    if win_detected and (current_time - last_detection_time['win']) > COOLDOWN_SECONDS:
        timestamp = extract_timestamp(frame, digit_templates)
        print(f"🏆 YOU WIN detected! Time: {timestamp} (confidence: {win_conf:.2f})")
        last_detection_time['win'] = current_time
    
    if record_detected and (current_time - last_detection_time['record']) > COOLDOWN_SECONDS:
        timestamp = extract_timestamp(frame, digit_templates)
        print(f"⭐ NEW RECORD detected! Time: {timestamp} (confidence: {record_conf:.2f})")
        last_detection_time['record'] = current_time

    if finish_detected and (current_time - last_detection_time['finish']) > COOLDOWN_SECONDS:
        timestamp = extract_timestamp(frame, digit_templates)
        print(f"🏁 FINISH detected! Time: {timestamp} (confidence: {finish_conf:.2f})")
        last_detection_time['finish'] = current_time
    
    # Draw rectangle on frame to visualize ROI
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
    
    # Display frame
    cv2.imshow('Game Feed', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()