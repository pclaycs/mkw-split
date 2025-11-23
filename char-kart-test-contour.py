import cv2
import numpy as np
from utils import find_obs_camera

# Your character and kart lists
KNOWN_CHARACTERS = [
    "Mario",
    "Luigi",
    "Peach",
    "Daisy",
    "Bowser",
    "Koopa Troopa",
    "Toad",
    "Yoshi",
    "Wario",
    "Waluigi",
    "Rosalina",
    "Pauline",
    "Birdo",
    "Lakitu",
    "Baby Mario",
    "Baby Luigi",
    "Toadette",
    "King Boo",
    "Shy Guy",
    "Bowser Jr.",
    "Baby Peach",
    "Baby Daisy",
    "Baby Rosalina",
    "Donkey Kong",
    "Dry Bones",
    "Goomba",
    "Nabbit",
    "Piranha Plant",
    "Spike",
    "Wiggler",
    "Cataquack",
    "Sidestepper",
    "Hammer Bro",
    "Monty Mole",
    "Cheep Cheep",
    "Pianta",
    "Rosalina (Touring)",
    "Yoshi (Touring)",
    "Mario (Touring)",
    "Luigi (Touring)",
    "Baby Peach (Touring)",
    "Koopa Troopa (Runner)",
    "Rocky Wrench",
    "Baby Daisy (Touring)",
    "Peach (Touring)",
    "Daisy (Touring)",
    "Baby Rosalina (Touring)",
    "Conkdor",
    "Wario (Pro Racer)",
    "Yoshi (Pro Racer)",
    "Mario (Pro Racer)",
    "Luigi (Pro Racer)",
    "Toad (Pro Racer)",
    "Waluigi (Pro Racer)",
    "Rosalina (Pro Racer)",
    "Toadette (Pro Racer)",
    "Peach (Pro Racer)",
    "Daisy (Pro Racer)",
    "Koopa Troopa (Pro Racer)",
    "Bowser (Pro Racer)",
    "Shy Guy (Pit Crew)",
    "Baby Rosalina (Pro Racer)",
    "Baby Mario (Pro Racer)",
    "Baby Luigi (Pro Racer)",
    "Bowser Jr. (Pro Racer)",
    "Lakitu (Pit Crew)",
    "Mario (Mechanic)",
    "Birdo (Pro Racer)",
    "Baby Peach (Pro Racer)",
    "Baby Daisy (Pro Racer)",
    "King Boo (Pro Racer)",
    "Luigi (Mechanic)",
    "Waluigi (Wampire)",
    "Wario (Oasis)",
    "Mario (Dune Rider)",
    "Luigi (Oasis)",
    "Yoshi (Aristocrat)",
    "Swoop",
    "Toadette (Conductor)",
    "King Boo (Aristocrat)",
    "Daisy (Oasis)",
    "Pokey",
    "Peepa",
    "Toad (Engineer)",
    "Waluigi (Mariachi)",
    "Wario (Wicked Wasp)",
    "Mario (Cowboy)",
    "Luigi (Farmer)",
    "Stingby",
    "Fish Bone",
    "Coin Coffer",
    "Toadette (Soft Server)",
    "Peach (Farmer)",
    "Cow",
    "Yoshi (Soft Server)",
    "Bowser (Supercharged)",
    "Rosalina (Aurora)",
    "Wario (Biker)",
    "Mario (Sightseeing)",
    "Peach (Sightseeing)",
    "Waluigi (Biker)",
    "Shy Guy (Slope Styler)",
    "Snowman",
    "Bowser (Biker)",
    "Toad (Burger Bud)",
    "Yoshi (Biker)",
    "Bowser Jr. (Biker Jr.)",
    "Penguin",
    "Daisy (Swimwear)",
    "Baby Peach (Sailor)",
    "Mario (Aviator)",
    "Peach (Aviator)",
    "Baby Daisy (Sailor)",
    "Yoshi (Swimwear)",
    "Baby Mario (Swimwear)",
    "Baby Rosalina (Sailor)",
    "Wario (Pirate)",
    "King Boo (Pirate)",
    "Koopa Troopa (Sailor)",
    "Dolphin",
    "Baby Peach (Explorer)",
    "Lakitu (Fisherman)",
    "Mario (Happi)",
    "Luigi (Happi)",
    "Para-Biddybud",
    "Baby Daisy (Explorer)",
    "Baby Rosalina (Explorer)",
    "Toad (Explorer)",
    "Peach (Yukata)",
    "Yoshi (Matsuri)",
    "Toadette (Explorer)",
    "Bowser Jr. (Explorer)",
    "Koopa Troopa (All-Terrain)",
    "Donkey Kong (All-Terrain)",
    "Mario (All-Terrain)",
    "Luigi (All-Terrain)",
    "Bowser (All-Terrain)",
    "Wario (Road Ruffian)",
    "Waluigi (Road Ruffian)",
    "Pauline (Aero)",
    "Peach (Aero)",
    "Daisy (Aero)",
    "Rosalina (Aero)",
    "Chargin' Chuck",
    "Wario (Work Crew)",
    "Baby Mario (Work Crew)",
    "Luigi (Gondolier)",
    "Peach (Vacation)",
    "Baby Luigi (Work Crew)",
    "Koopa Troopa (Work Crew)",
    "Yoshi (Food Slinger)",
    "Birdo (Vacation)",
    "Daisy (Vacation)"
]

KNOWN_KARTS = [
    "Standard Kart",
    "Rally Kart",
    "Standard Bike",
    "Rally Bike",
    "Mach Rocket",
    "Cute Scoot",
    "Baby Blooper",
    "Plushbuggy",
    "Zoom Buggy",
    "Chargin' Truck",
    "Hyper Pipe",
    "Funky Dorrie",
    "Junkyard Hog",
    "Tune Thumper",
    "Ribbit Revster",
    "Hot Rod",
    "Roadster Royale",
    "B Dasher",
    "W-Twin Chopper",
    "Lobster Roller",
    "Stellar Sled",
    "Dread Sled",
    "Rally Romper",
    "Buggybud",
    "Reel Racer",
    "Bumble V",
    "Fin Twin",
    "R.O.B. H.O.G.",
    "Blastronaut III",
    "Dolphin Dasher",
    "Cloud 9",
    "Carpet Flyer",
    "Big Horn",
    "Li'l Dumpy",
    "Loco Moto",
    "Mecha Trike",
    "Bowser Bruiser",
    "Rallygator",
    "Billdozer",
    "Pipe Frame",
]

def load_templates():
    """Load all character and kart templates"""
    templates = {
        'characters': {},
        'karts': {}
    }
    
    # Load character templates
    for char in KNOWN_CHARACTERS:
        filename = char.replace('(', '').replace(')', '').replace('.', '').replace(' ', '_').lower() + '.png'
        img = cv2.imread(f'images/characters/{filename}', cv2.IMREAD_GRAYSCALE)
        if img is not None:
            templates['characters'][char] = img
    
    # Load kart templates
    for kart in KNOWN_KARTS:
        filename = kart.replace('(', '').replace(')', '').replace('.', '').replace(' ', '_').lower() + '.png'
        img = cv2.imread(f'images/karts/{filename}', cv2.IMREAD_GRAYSCALE)
        if img is not None:
            templates['karts'][kart] = img
    
    return templates

def detect_character_or_kart(frame, templates, char_roi_coords, kart_roi_coords, threshold=0.9):
    """Detect both character and kart in one pass"""
    char_x1, char_y1, char_x2, char_y2 = char_roi_coords
    kart_x1, kart_y1, kart_x2, kart_y2 = kart_roi_coords
    
    # Extract and preprocess character ROI
    char_roi = frame[char_y1:char_y2, char_x1:char_x2]
    char_gray = cv2.cvtColor(char_roi, cv2.COLOR_BGR2GRAY)
    _, char_thresh = cv2.threshold(char_gray, 170, 255, cv2.THRESH_BINARY)
    
    # Extract and preprocess kart ROI
    kart_roi = frame[kart_y1:kart_y2, kart_x1:kart_x2]
    kart_gray = cv2.cvtColor(kart_roi, cv2.COLOR_BGR2GRAY)
    _, kart_thresh = cv2.threshold(kart_gray, 170, 255, cv2.THRESH_BINARY)
    
    # Match character
    best_char = None
    best_char_score = 0
    
    for name, template in templates['characters'].items():
        if template.shape[0] > char_thresh.shape[0] or template.shape[1] > char_thresh.shape[1]:
            continue
        
        result = cv2.matchTemplate(char_thresh, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > best_char_score:
            best_char_score = max_val
            best_char = name
    
    # Match kart
    best_kart = None
    best_kart_score = 0
    
    for name, template in templates['karts'].items():
        if template.shape[0] > kart_thresh.shape[0] or template.shape[1] > kart_thresh.shape[1]:
            continue
        
        result = cv2.matchTemplate(kart_thresh, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > best_kart_score:
            best_kart_score = max_val
            best_kart = name
    
    # Only return if above threshold
    char_result = (best_char, best_char_score) if best_char_score >= threshold else (None, best_char_score)
    kart_result = (best_kart, best_kart_score) if best_kart_score >= threshold else (None, best_kart_score)
    
    return char_result, kart_result

# Load templates
print("Loading templates...")
templates = load_templates()
print(f"Loaded {len(templates['characters'])} character templates")
print(f"Loaded {len(templates['karts'])} kart templates")

# Open camera
camera_index = find_obs_camera()
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("Camera opened. Press 'q' to quit\n")

# ROI coordinates - adjusted to match preprocessing crops
CHAR_X1, CHAR_Y1 = 1210, 828  # 808 + 20 (top crop)
CHAR_X2, CHAR_Y2 = 1770, 960  # 970 - 10 (bottom crop)

KART_X1, KART_Y1 = 1240, 828  # 1210 + 30, 808 + 20
KART_X2, KART_Y2 = 1740, 950  # 1770 - 30, 970 - 20

# Track last detected
last_character = None
last_kart = None

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Detect both at once
    char_result, kart_result = detect_character_or_kart(
        frame, templates,
        (CHAR_X1, CHAR_Y1, CHAR_X2, CHAR_Y2),
        (KART_X1, KART_Y1, KART_X2, KART_Y2)
    )
    
    char_name, char_conf = char_result
    kart_name, kart_conf = kart_result
    
    # Update if changed
    if char_name and char_name != last_character:
        last_character = char_name
        print(f"🎮 Character: {last_character} (confidence: {char_conf:.3f})")
    
    if kart_name and kart_name != last_kart:
        last_kart = kart_name
        print(f"🏎️ Kart: {last_kart} (confidence: {kart_conf:.3f})")
    
    # Draw ROI rectangles
    cv2.rectangle(frame, (CHAR_X1, CHAR_Y1), (CHAR_X2, CHAR_Y2), (0, 255, 255), 2)
    cv2.rectangle(frame, (KART_X1, KART_Y1), (KART_X2, KART_Y2), (255, 0, 255), 2)
    
    # Display current selections
    cv2.putText(frame, f"Character: {last_character or 'None'}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Kart: {last_kart or 'None'}", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Character/Kart Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\nFinal selections:")
print(f"Character: {last_character}")
print(f"Kart: {last_kart}")

cap.release()
cv2.destroyAllWindows()