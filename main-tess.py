import cv2
import numpy as np
import pytesseract
from pygrabber.dshow_graph import FilterGraph
import re
from utils import find_obs_camera

# Set Tesseract path (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def validate_timestamp(text):
    """Check if text matches timestamp format"""
    # Pattern: M:SS.mmm or MM:SS.mmm
    pattern = r'^\d{1,2}:\d{2}\.\d{3}$'
    return re.match(pattern, text) is not None

camera_index = find_obs_camera()

# Open camera and set resolution
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Verify resolution
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {actual_width}x{actual_height}")

# ROI coordinates
roi_x, roi_y, roi_w, roi_h = 1556, 56, 308, 50

# Tesseract config - digits, colon, and period only
custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:.'

print("Reading from ROI... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Extract ROI
    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    
    # Preprocess the ROI
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh_roi = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)
    
    # Optional: Apply morphological closing to connect broken characters
    kernel = np.ones((2,2), np.uint8)
    thresh_roi = cv2.morphologyEx(thresh_roi, cv2.MORPH_CLOSE, kernel)
    
    # Run Tesseract OCR
    scale_factor = 3
    thresh_roi_large = cv2.resize(thresh_roi, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    text = pytesseract.image_to_string(thresh_roi_large, config=custom_config)
    text = text.strip()
    
    # Only print if we got text
    if text:
        is_valid = validate_timestamp(text)
        status = "✓" if is_valid else "?"
        print(f"{status} '{text}'")
    
    # Display both for comparison
    cv2.imshow('ROI Original', roi)
    cv2.imshow('ROI Processed', thresh_roi)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()