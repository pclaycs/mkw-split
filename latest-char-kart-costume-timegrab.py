import cv2
import numpy as np
from utils import find_obs_camera
import time
import os

last_detection_time = {
    'lose': 0,
    'win': 0,
    'record': 0,
    'finish': 0,
    'lap': 0
}
COOLDOWN_SECONDS = 5

# Track yellow timestamp state
was_yellow = False

# Track selected character/kart/costume
selected_character = None
selected_costume = None
selected_kart = None

KNOWN_COSTUMES = {
    "Baby Daisy": ['Touring', 'Pro Racer', 'Sailor', 'Explorer'],
    "Baby Luigi": ['Pro Racer', 'Work Crew'],
    "Baby Mario": ['Pro Racer', 'Swimwear', 'Work Crew'],
    "Baby Peach": ['Touring', 'Pro Racer', 'Sailor', 'Explorer'],
    "Baby Rosalina": ['Touring', 'Pro Racer', 'Sailor', 'Explorer'],
    "Birdo": ['Pro Racer', 'Vacation'],
    "Bowser": ['Pro Racer', 'Supercharged', 'Biker', 'All-Terrain'],
    "Bowser Jr.": ['Pro Racer', 'Biker Jr.', 'Explorer'],
    "Cataquack": [],
    "Chargin' Chuck": [],
    "Cheep Cheep": [],
    "Coin Coffer": [],
    "Conkdor": [],
    "Cow": [],
    "Daisy": ['Touring', 'Pro Racer', 'Oasis', 'Swimwear', 'Aero', 'Vacation'],
    "Dolphin": [],
    "Donkey Kong": ['All-Terrain'],
    "Dry Bones": [],
    "Fish Bone": [],
    "Goomba": [],
    "Hammer Bro": [],
    "King Boo": ['Pro Racer', 'Aristocrat', 'Pirate'],
    "Koopa Troopa": ['Runner', 'Pro Racer', 'Sailor', 'All-Terrain', 'Work Crew'],
    "Lakitu": ['Pit Crew', 'Fisherman'],
    "Luigi": ['Touring', 'Pro Racer', 'Mechanic', 'Oasis', 'Farmer', 'Happi', 'All-Terrain', 'Gondolier'],
    "Mario": ['Touring', 'Pro Racer', 'Mechanic', 'Dune Rider', 'Cowboy', 'Sightseeing', 'Aviator', 'Happi', 'All-Terrain'],
    "Monty Mole": [],
    "Nabbit": [],
    "Para-Biddybud": [],
    "Pauline": ['Aero'],
    "Peach": ['Touring', 'Pro Racer', 'Farmer', 'Sightseeing', 'Aviator', 'Yukata', 'Aero', 'Vacation'],
    "Peepa": [],
    "Penguin": [],
    "Pianta": [],
    "Piranha Plant": [],
    "Pokey": [],
    "Rocky Wrench": [],
    "Rosalina": ['Touring', 'Pro Racer', 'Aurora', 'Aero'],
    "Shy Guy": ['Pit Crew', 'Slope Styler'],
    "Sidestepper": [],
    "Snowman": [],
    "Spike": [],
    "Stingby": [],
    "Swoop": [],
    "Toad": ['Pro Racer', 'Engineer', 'Burger Bud', 'Explorer'],
    "Toadette": ['Pro Racer', 'Conductor', 'Soft Server', 'Explorer'],
    "Waluigi": ['Pro Racer', 'Wampire', 'Mariachi', 'Biker', 'Road Ruffian'],
    "Wario": ['Pro Racer', 'Oasis', 'Wicked Wasp', 'Biker', 'Pirate', 'Road Ruffian', 'Work Crew'],
    "Wiggler": [],
    "Yoshi": ['Touring', 'Pro Racer', 'Aristocrat', 'Soft Server', 'Biker', 'Swimwear', 'Matsuri', 'Food Slinger'],
}

# ROI definitions
CHAR_NAME_ROI = (1210, 830, 1770, 894)
COSTUME_ROI = (1210, 916, 1770, 958)
KART_NAME_ROI = (1240, 830, 1740, 894)
SCREEN_INDICATOR_ROI = (1360, 1024, 1920, 1080)

def detect_with_mask(frame, template, mask, threshold=0.7):
    """Template matching with mask - single scale only"""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    result = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED, mask=mask)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    
    if max_val >= threshold:
        return True, max_val
    
    return False, 0.0

def detect_template_in_roi(roi, template, threshold=0.9):
    """Match template in ROI with high confidence threshold"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    
    if template.shape[0] > thresh.shape[0] or template.shape[1] > thresh.shape[1]:
        return False, 0.0
    
    result = cv2.matchTemplate(thresh, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    
    return max_val >= threshold, max_val

def load_template_with_mask(path):
    """Load a PNG with transparency and extract mask"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    
    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        _, mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
        return bgr, mask
    else:
        mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
        return img, mask

def load_selection_templates():
    """Load character, costume, and kart templates"""
    templates = {
        'characters': {},
        'costumes': {},
        'karts': {},
        'screens': {}
    }
    
    # Load character templates
    char_dir = 'images/characters'
    if os.path.exists(char_dir):
        for filename in os.listdir(char_dir):
            if filename.endswith('.png'):
                name = filename[:-4].replace('_', ' ').title()
                img = cv2.imread(os.path.join(char_dir, filename), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    templates['characters'][name] = img
    
    # Load costume templates
    costume_dir = 'images/costumes'
    if os.path.exists(costume_dir):
        for filename in os.listdir(costume_dir):
            if filename.endswith('.png'):
                name = filename[:-4].replace('_', ' ').title()
                img = cv2.imread(os.path.join(costume_dir, filename), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    templates['costumes'][name] = img
    
    # Load kart templates
    kart_dir = 'images/karts'
    if os.path.exists(kart_dir):
        for filename in os.listdir(kart_dir):
            if filename.endswith('.png'):
                name = filename[:-4].replace('_', ' ').title()
                img = cv2.imread(os.path.join(kart_dir, filename), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    templates['karts'][name] = img
    
    # Load screen indicators
    char_screen = cv2.imread('images/screens/character_screen.png', cv2.IMREAD_GRAYSCALE)
    kart_screen = cv2.imread('images/screens/kart_screen.png', cv2.IMREAD_GRAYSCALE)
    if char_screen is not None:
        templates['screens']['character'] = char_screen
    if kart_screen is not None:
        templates['screens']['kart'] = kart_screen
    
    return templates

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

def match_best_template(roi, templates, threshold=0.9):
    """Find best matching template above threshold"""
    best_name = None
    best_score = 0
    
    for name, template in templates.items():
        detected, score = detect_template_in_roi(roi, template, threshold=0)
        if score > best_score:
            best_score = score
            best_name = name
    
    if best_score >= threshold:
        return best_name, best_score
    return None, best_score

def is_timestamp_yellow(frame):
    """Check if timestamp has turned yellow (lap complete)"""
    timestamp_roi = frame[54:107, 1550:1864]
    hsv = cv2.cvtColor(timestamp_roi, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 80, 180])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_ratio = np.count_nonzero(yellow_mask) / yellow_mask.size
    return yellow_ratio > 0.2

def extract_timestamp(frame, templates):
    """Extract timestamp from frame"""
    timestamp_roi = frame[54:107, 1550:1864]
    gray = cv2.cvtColor(timestamp_roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    digits = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 15 or h < 20:
            continue
        char_roi = thresh[y:y+h, x:x+w]
        char, conf = recognize_character(char_roi, templates)
        if char and conf > 0.3:
            digits.append(char)
    
    if len(digits) == 6:
        timestamp = f"{digits[0]}:{digits[1]}{digits[2]}.{digits[3]}{digits[4]}{digits[5]}"
    else:
        timestamp = ''.join(digits)
    
    return timestamp

# Load templates
print("Loading templates...")
you_lose_template, you_lose_mask = load_template_with_mask('images/crop-you-lose-nobg.png')
you_win_template, you_win_mask = load_template_with_mask('images/crop-you-win-nobg.png')
new_record_template, new_record_mask = load_template_with_mask('images/crop-new-record-nobg.png')
finish_template, finish_mask = load_template_with_mask('images/crop-finish-nobg.png')

digit_templates = load_digit_templates()
selection_templates = load_selection_templates()

print(f"Loaded {len(selection_templates['characters'])} characters, "
      f"{len(selection_templates['costumes'])} costumes, "
      f"{len(selection_templates['karts'])} karts")
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
    
    current_time = time.time()
    
    # Check for screen indicators
    screen_indicator_roi = frame[SCREEN_INDICATOR_ROI[1]:SCREEN_INDICATOR_ROI[3], 
                                   SCREEN_INDICATOR_ROI[0]:SCREEN_INDICATOR_ROI[2]]
    
    char_screen_detected, _ = detect_template_in_roi(screen_indicator_roi, 
                                                       selection_templates['screens'].get('character'), 
                                                       threshold=0.9)
    kart_screen_detected, _ = detect_template_in_roi(screen_indicator_roi, 
                                                       selection_templates['screens'].get('kart'), 
                                                       threshold=0.9)
    
    # Character screen detection
    if char_screen_detected:
        char_name_roi = frame[CHAR_NAME_ROI[1]:CHAR_NAME_ROI[3], CHAR_NAME_ROI[0]:CHAR_NAME_ROI[2]]
        char_name, char_conf = match_best_template(char_name_roi, selection_templates['characters'])
        
        if char_name and char_name != selected_character:
            selected_character = char_name
            selected_costume = None  # Reset costume when character changes
            print(f"🎮 Character: {selected_character} (confidence: {char_conf:.3f})")
        
        # Only check for costume if we have a selected character AND on character screen
        if selected_character:
            # Check for costume if character has costumes
            if selected_character in KNOWN_COSTUMES and len(KNOWN_COSTUMES[selected_character]) > 0:
                costume_roi = frame[COSTUME_ROI[1]:COSTUME_ROI[3], COSTUME_ROI[0]:COSTUME_ROI[2]]
                
                # Only check costumes for this character
                relevant_costumes = {name: template for name, template in selection_templates['costumes'].items() 
                                    if name in KNOWN_COSTUMES[selected_character]}
                
                costume_name, costume_conf = match_best_template(costume_roi, relevant_costumes)
                
                # Update if we found a costume match
                if costume_name and costume_name != selected_costume:
                    selected_costume = costume_name
                    print(f"👔 Costume: {selected_costume} (confidence: {costume_conf:.3f})")
                # Reset to base ONLY if we detect character name but NO costume with decent confidence
                # AND we detected the character itself with high confidence (confirming we're stable on screen)
                elif costume_conf < 0.5 and char_conf > 0.95 and selected_costume is not None:
                    selected_costume = None
                    print(f"👔 Costume: Base (no costume)")
            else:
                # Character has no costumes available - only reset if character detection is stable
                if selected_costume is not None and char_conf > 0.95:
                    selected_costume = None
                    print(f"👔 Costume: Base (no costume)")
    
    # Kart screen detection
    if kart_screen_detected:
        kart_name_roi = frame[KART_NAME_ROI[1]:KART_NAME_ROI[3], KART_NAME_ROI[0]:KART_NAME_ROI[2]]
        kart_name, kart_conf = match_best_template(kart_name_roi, selection_templates['karts'])
        
        if kart_name and kart_name != selected_kart:
            selected_kart = kart_name
            print(f"🏎️ Kart: {selected_kart} (confidence: {kart_conf:.3f})")
    
    # Extract ROI for result detection
    result_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # Check for each result screen
    lose_detected, lose_conf = detect_with_mask(result_roi, you_lose_template, you_lose_mask)
    win_detected, win_conf = detect_with_mask(result_roi, you_win_template, you_win_mask)
    record_detected, record_conf = detect_with_mask(result_roi, new_record_template, new_record_mask)
    finish_detected, finish_conf = detect_with_mask(result_roi, finish_template, finish_mask)
    
    # Check for yellow timestamp (lap complete)
    yellow_detected = is_timestamp_yellow(frame)
    
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
    
    # Handle yellow timestamp state transition
    if yellow_detected and not was_yellow:
        if (current_time - last_detection_time['lap']) > COOLDOWN_SECONDS:
            was_yellow = True
    elif not yellow_detected and was_yellow:
        timestamp = extract_timestamp(frame, digit_templates)
        print(f"🔃 LAP COMPLETE! Time: {timestamp}")
        last_detection_time['lap'] = current_time
        was_yellow = False
    
    # Draw rectangles on frame to visualize ROIs
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)

    # Draw character/kart selection ROIs when on those screens
    if char_screen_detected:
        # Character name ROI (cyan)
        cv2.rectangle(frame, (CHAR_NAME_ROI[0], CHAR_NAME_ROI[1]), 
                    (CHAR_NAME_ROI[2], CHAR_NAME_ROI[3]), (255, 255, 0), 2)
        
        # Costume ROI (magenta) - only if character has costumes
        if selected_character and selected_character in KNOWN_COSTUMES and len(KNOWN_COSTUMES[selected_character]) > 0:
            cv2.rectangle(frame, (COSTUME_ROI[0], COSTUME_ROI[1]), 
                        (COSTUME_ROI[2], COSTUME_ROI[3]), (255, 0, 255), 2)
        
        # Screen indicator ROI (yellow)
        cv2.rectangle(frame, (SCREEN_INDICATOR_ROI[0], SCREEN_INDICATOR_ROI[1]), 
                    (SCREEN_INDICATOR_ROI[2], SCREEN_INDICATOR_ROI[3]), (0, 255, 255), 2)

    if kart_screen_detected:
        # Kart name ROI (cyan)
        cv2.rectangle(frame, (KART_NAME_ROI[0], KART_NAME_ROI[1]), 
                    (KART_NAME_ROI[2], KART_NAME_ROI[3]), (255, 255, 0), 2)
        
        # Screen indicator ROI (yellow)
        cv2.rectangle(frame, (SCREEN_INDICATOR_ROI[0], SCREEN_INDICATOR_ROI[1]), 
                    (SCREEN_INDICATOR_ROI[2], SCREEN_INDICATOR_ROI[3]), (0, 255, 255), 2)

    # Display current selections
    cv2.putText(frame, f"Char: {selected_character or 'None'}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Costume: {selected_costume or 'None'}", (50, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Kart: {selected_kart or 'None'}", (50, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Game Feed', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\nFinal selections:")
print(f"Character: {selected_character}")
print(f"Costume: {selected_costume}")
print(f"Kart: {selected_kart}")

cap.release()
cv2.destroyAllWindows()