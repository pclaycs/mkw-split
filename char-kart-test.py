import cv2
import numpy as np
from utils import find_obs_camera
from paddleocr import PaddleOCR
from difflib import get_close_matches

KNOWN_CHARACTERS = [
    "Mario",
    "Luigi",
    "Peach",
    "Daisy",
    "Yoshi",
    "Toad",
    "Koopa Troopa",
    "Bowser",
    "Wario",
    "Waluigi",
    "Rosalina",
    "Pauline",
    "Baby Mario",
    "Baby Luigi",
    "Baby Peach",
    "Baby Daisy",
    "Lakitu",
    "Toadette",
    "Bowser Jr.",
    "Baby Rosalina",
    "Birdo",
    "King Boo",
    "Shy Guy",
    "Donkey Kong",
    "Nabbit",
    "Piranha Plant",
    "Hammer Bro",
    "Monty Mole",
    "Goomba",
    "Spike",
    "Sidestepper",
    "Cheep Cheep",
    "Dry Bones",
    "Wiggler",
    "Cataquack",
    "Pianta",
    "Mario (Touring)",
    "Luigi (Touring)",
    "Peach (Touring)",
    "Daisy (Touring)",
    "Yoshi (Touring)",
    "Baby Peach (Touring)",
    "Baby Daisy (Touring)",
    "Baby Rosalina (Touring)",
    "Rosalina (Touring)",
    "Koopa Troopa (Runner)",
    "Rocky Wrench",
    "Conkdor",
    "Mario (Pro Racer)",
    "Luigi (Pro Racer)",
    "Peach (Pro Racer)",
    "Daisy (Pro Racer)",
    "Yoshi (Pro Racer)",
    "Toad (Pro Racer)",
    "Toadette (Pro Racer)",
    "Koopa Troopa (Pro Racer)",
    "Wario (Pro Racer)",
    "Waluigi (Pro Racer)",
    "Rosalina (Pro Racer)",
    "Bowser (Pro Racer)",
    "Baby Mario (Pro Racer)",
    "Baby Luigi (Pro Racer)",
    "Baby Peach (Pro Racer)",
    "Baby Daisy (Pro Racer)",
    "Baby Rosalina (Pro Racer)",
    "Bowser Jr. (Pro Racer)",
    "Birdo (Pro Racer)",
    "King Boo (Pro Racer)",
    "Shy Guy (Pit Crew)",
    "Lakitu (Pit Crew)",
    "Mario (Mechanic)",
    "Luigi (Mechanic)",
    "Mario (Dune Rider)",
    "Luigi (Oasis)",
    "Daisy (Oasis)",
    "Pokey",
    "Wario (Oasis)",
    "Yoshi (Aristocrat)",
    "King Boo (Aristocrat)",
    "Peepa",
    "Waluigi (Wampire)",
    "Swoop",
    "Toadette (Conductor)",
    "Toad (Engineer)",
    "Mario (Cowboy)",
    "Luigi (Farmer)",
    "Peach (Farmer)",
    "Cow",
    "Wario (Wicked Wasp)",
    "Stingby",
    "Toadette (Soft Server)",
    "Yoshi (Soft Server)",
    "Waluigi (Mariachi)",
    "Fish Bone",
    "Coin Coffer",
    "Bowser (Supercharged)",
    "Mario (Sightseeing)",
    "Peach (Sightseeing)",
    "Toad (Burger Bud)",
    "Yoshi (Biker)",
    "Wario (Biker)",
    "Waluigi (Biker)",
    "Bowser (Biker)",
    "Bowser Jr. (Biker Jr.)",
    "Rosalina (Aurora)",
    "Shy Guy (Slope Styler)",
    "Snowman",
    "Penguin",
    "Mario (Aviator)",
    "Peach (Aviator)",
    "Wario (Pirate)",
    "King Boo (Pirate)",
    "Baby Peach (Sailor)",
    "Baby Daisy (Sailor)",
    "Baby Rosalina (Sailor)",
    "Koopa Troopa (Sailor)",
    "Daisy (Swimwear)",
    "Yoshi (Swimwear)",
    "Baby Mario (Swimwear)",
    "Dolphin",
    "Mario (Happi)",
    "Luigi (Happi)",
    "Peach (Yukata)",
    "Yoshi (Matsuri)",
    "Lakitu (Fisherman)",
    "Para-Biddybud",
    "Toad (Explorer)",
    "Toadette (Explorer)",
    "Baby Peach (Explorer)",
    "Baby Daisy (Explorer)",
    "Baby Rosalina (Explorer)",
    "Bowser Jr. (Explorer)",
    "Mario (All-Terrain)",
    "Luigi (All-Terrain)",
    "Peach (Aero)",
    "Daisy (Aero)",
    "Donkey Kong (All-Terrain)",
    "Bowser (All-Terrain)",
    "Pauline (Aero)",
    "Rosalina (Aero)",
    "Koopa Troopa (All-Terrain)",
    "Wario (Road Ruffian)",
    "Waluigi (Road Ruffian)",
    "Chargin' Chuck",
    "Luigi (Gondolier)",
    "Peach (Vacation)",
    "Daisy (Vacation)",
    "Baby Mario (Work Crew)",
    "Baby Luigi (Work Crew)",
    "Birdo (Vacation)",
    "Wario (Work Crew)",
    "Koopa Troopa (Work Crew)",
    "Yoshi (Food Slinger)",
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

def extract_text_from_roi(frame, ocr):
    """Extract character/kart name and optional type from ROI"""
    # Full ROI
    roi = frame[808:970, 1210:1770]
    
    # Split into name (top ~60%) and type (bottom ~40%)
    roi_height = roi.shape[0]
    split_point = int(roi_height * 0.6)
    
    name_roi = roi[0:split_point, :]
    type_roi = roi[split_point:, :]
    
    # OCR both regions
    name_result = ocr.predict(name_roi)
    type_result = ocr.predict(type_roi)
    
    # Extract text
    name_text = ""
    type_text = ""
    
    if name_result and name_result[0]:
        rec_texts = name_result[0].get('rec_texts', [])
        if rec_texts:
            name_text = ' '.join(rec_texts).strip()
    
    if type_result and type_result[0]:
        rec_texts = type_result[0].get('rec_texts', [])
        if rec_texts:
            type_text = ' '.join(rec_texts).strip()
    
    return name_text, type_text

def match_character_or_kart(name_text, type_text):
    """Match against known characters or karts using fuzzy matching"""
    # Try matching as character with skin (Name + Type)
    if type_text:
        full_text = f"{name_text} ({type_text})"
        char_matches = get_close_matches(full_text, KNOWN_CHARACTERS, n=1, cutoff=0.6)
        if char_matches:
            return "character", char_matches[0]
        
        # Try without parentheses in case OCR missed them
        char_matches = get_close_matches(name_text, KNOWN_CHARACTERS, n=1, cutoff=0.6)
        if char_matches:
            return "character", char_matches[0]
    
    # Try matching as character without skin
    char_matches = get_close_matches(name_text, KNOWN_CHARACTERS, n=1, cutoff=0.6)
    if char_matches:
        return "character", char_matches[0]
    
    # Try matching as kart
    kart_matches = get_close_matches(name_text, KNOWN_KARTS, n=1, cutoff=0.6)
    if kart_matches:
        return "kart", kart_matches[0]
    
    # No match found
    return None, f"{name_text} ({type_text})" if type_text else name_text

# Initialize PaddleOCR
print("Loading OCR...")
ocr = PaddleOCR(use_textline_orientation=True, lang='en')
print("OCR loaded!")

# Open camera
camera_index = find_obs_camera()
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# ROI coordinates
ROI_X1, ROI_Y1 = 1210, 808
ROI_X2, ROI_Y2 = 1770, 970

# Track last detected character and kart
last_character = None
last_kart = None

print("Detecting character/kart selection... Press 'q' to quit")
print(f"Current character: {last_character}")
print(f"Current kart: {last_kart}")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Extract text from ROI
    name_text, type_text = extract_text_from_roi(frame, ocr)
    
    # Only process if we detected some text
    if name_text:
        # Match against known lists
        detection_type, matched_text = match_character_or_kart(name_text, type_text)
        
        if detection_type == "character" and matched_text != last_character:
            last_character = matched_text
            print(f"\n🎮 Character selected: {last_character}")
        elif detection_type == "kart" and matched_text != last_kart:
            last_kart = matched_text
            print(f"\n🏎️ Kart selected: {last_kart}")
        elif detection_type is None:
            # Unknown text detected - show what we got
            print(f"❓ Unknown: '{matched_text}' (Raw: name='{name_text}', type='{type_text}')")
    
    # Draw ROI rectangle for visualization
    cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 2)
    
    # Display current selections on frame
    cv2.putText(frame, f"Character: {last_character}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Kart: {last_kart}", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display frame
    cv2.imshow('Character/Kart Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\nFinal selections:")
print(f"Character: {last_character}")
print(f"Kart: {last_kart}")

cap.release()
cv2.destroyAllWindows()