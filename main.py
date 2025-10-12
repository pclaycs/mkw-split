import cv2
from paddleocr import PaddleOCR
from pygrabber.dshow_graph import FilterGraph
import numpy as np
import pytesseract
from utils import find_obs_camera

ocr = PaddleOCR(use_textline_orientation=True, lang='en')

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
roi_x, roi_y, roi_w, roi_h = 1548, 52, 324, 58

# Allowed characters for timestamp
ALLOWED_CHARS = set('0123456789:.')

def filter_timestamp(text):
    """Filter text to only allowed characters and validate format"""
    # Remove any characters not in allowed set
    filtered = ''.join(c for c in text if c in ALLOWED_CHARS)
    return filtered

def validate_timestamp(text):
    """Check if text matches timestamp format"""
    import re
    # Pattern: M:SS.mmm or MM:SS.mmm
    pattern = r'^\d{1,2}:\d{2}\.\d{3}$'
    return re.match(pattern, text) is not None

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

    # Convert back to BGR (3 channels) for PaddleOCR
    thresh_roi_bgr = cv2.cvtColor(thresh_roi, cv2.COLOR_GRAY2BGR)

    # Run OCR on preprocessed ROI
    result = ocr.predict(thresh_roi_bgr)
    
    # Print detected text - result is a list of dicts
    if result and len(result) > 0:
        page_result = result[0]  # First page
        
        if isinstance(page_result, dict):
            # Extract recognized texts and scores
            rec_texts = page_result.get('rec_texts', [])
            rec_scores = page_result.get('rec_scores', [])
            
            if rec_texts:
                for i, text in enumerate(rec_texts):
                    # Filter to allowed characters
                    filtered_text = filter_timestamp(text)
                    
                    if filtered_text:  # Only print if we got something
                        confidence = rec_scores[i] if i < len(rec_scores) else 0
                        
                        # Check if it's a valid timestamp
                        is_valid = validate_timestamp(filtered_text)
                        status = "✓" if is_valid else "?"
                        
                        print(f"{status} '{filtered_text}' (conf: {confidence:.2f}, raw: '{text}')")
    
    # Optional: Display the ROI for debugging
    cv2.imshow('ROI Original', roi)
    cv2.imshow('ROI Processed', thresh_roi)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()