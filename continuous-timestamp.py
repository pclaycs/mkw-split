import cv2
import numpy as np
from utils import find_obs_camera

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
    best_match = None
    best_score = 0
    
    for char, template in templates.items():
        resized_template = cv2.resize(template, (roi.shape[1], roi.shape[0]))
        result = cv2.matchTemplate(roi, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > best_score:
            best_score = max_val
            best_match = char
    
    return best_match, best_score

def extract_timestamp(frame, templates):
    """Extract timestamp from frame"""
    timestamp_roi = frame[54:107, 1550:1864]
    
    gray = cv2.cvtColor(timestamp_roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    digits = []
    confidences = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        if w < 15 or h < 20:
            continue
        
        char_roi = thresh[y:y+h, x:x+w]
        char, conf = recognize_character(char_roi, templates)
        
        if char and conf > 0.3:
            digits.append(char)
            confidences.append(conf)
    
    if len(digits) == 6:
        timestamp = f"{digits[0]}:{digits[1]}{digits[2]}.{digits[3]}{digits[4]}{digits[5]}"
    else:
        timestamp = ''.join(digits)
    
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    return timestamp, avg_confidence, confidences

# Load digit templates
print("Loading templates...")
digit_templates = load_digit_templates()
print("Templates loaded!")

# Open camera
camera_index = find_obs_camera()
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("Reading timestamps... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Extract and print timestamp
    timestamp, avg_conf, individual_confs = extract_timestamp(frame, digit_templates)
    
    # Format individual confidences
    conf_str = ', '.join([f'{c:.2f}' for c in individual_confs])
    
    print(f"Time: {timestamp} | Avg confidence: {avg_conf:.2f} | Individual: [{conf_str}]")
    
    # Draw timestamp ROI for visualization
    cv2.rectangle(frame, (1550, 54), (1864, 107), (0, 255, 0), 2)
    
    # Display frame
    cv2.imshow('Timestamp Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()