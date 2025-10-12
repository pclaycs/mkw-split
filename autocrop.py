import cv2
import numpy as np
import os

def auto_crop_template(image_path, output_path, padding=2):
    """Automatically crop template to content with minimal padding"""
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"Could not load {image_path}")
        return
    
    print(f"\nProcessing {image_path}")
    print(f"  Original shape: {img.shape}")
    
    # Convert to grayscale
    if len(img.shape) == 3:
        if img.shape[2] == 4:  # Has alpha channel
            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply SAME threshold as we use for detection
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # DEBUG: Show the result
    cv2.imshow('Original', gray)
    cv2.imshow('Thresholded', thresh)
    cv2.waitKey(0)
    
    # Find contours of the content
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"  Found {len(contours)} contours")
    
    if not contours:
        print(f"No content found in {image_path}")
        return
    
    # Get bounding box of all content
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    print(f"  Bounding box: x={x}, y={y}, w={w}, h={h}")
    
    # Add padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(thresh.shape[1] - x, w + 2 * padding)
    h = min(thresh.shape[0] - y, h + 2 * padding)
    
    print(f"  After padding: x={x}, y={y}, w={w}, h={h}")
    
    # Crop the THRESHOLDED image (not original gray)
    cropped = thresh[y:y+h, x:x+w]
    
    # Save
    cv2.imwrite(output_path, cropped)
    print(f"Saved to {output_path} (size: {w}x{h})")

# Process all digit templates
os.makedirs('images/timestamps/cropped', exist_ok=True)

for i in range(10):
    auto_crop_template(f'images/timestamps/{i}.png', f'images/timestamps/cropped/{i}.png')

auto_crop_template('images/timestamps/colon.png', 'images/timestamps/cropped/colon.png')
auto_crop_template('images/timestamps/period.png', 'images/timestamps/cropped/period.png')

cv2.destroyAllWindows()
print("\nDone! Check the 'cropped' folder for results.")