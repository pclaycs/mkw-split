import cv2
import os
import numpy as np

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
SCREEN_INDICATOR_ROI = (1360, 1024, 1920, 1080)  # Bottom right, 560x56

def preprocess_roi(roi):
    """Apply preprocessing to ROI"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    return thresh

def sanitize_filename(name):
    """Convert name to safe filename"""
    return name.replace('(', '').replace(')', '').replace(' ', '_').replace('.', '').replace("'", '').lower()

def extract_screen_indicator(video_path, output_path):
    """Extract the bottom right screen indicator"""
    cap = cv2.VideoCapture(video_path)
    
    # Get first frame at 0.5s
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(0.5 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame from {video_path}")
        cap.release()
        return
    
    # Extract ROI
    x1, y1, x2, y2 = SCREEN_INDICATOR_ROI
    roi = frame[y1:y2, x1:x2]
    
    # Preprocess
    preprocessed = preprocess_roi(roi)
    
    # Save
    cv2.imwrite(output_path, preprocessed)
    print(f"Saved screen indicator: {output_path}")
    
    cap.release()

def extract_frames_at_intervals(video_path, items_list, roi, output_dir, interval=1.0, offset=0.5):
    """Extract frames at specific time intervals"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    os.makedirs(output_dir, exist_ok=True)
    
    x1, y1, x2, y2 = roi
    seen_items = set()
    
    for i, item_name in enumerate(items_list):
        # Skip if we've already saved this item
        filename = sanitize_filename(item_name)
        if filename in seen_items:
            print(f"Skipping duplicate: {item_name}")
            continue
        
        # Calculate timestamp: offset + (i * interval)
        timestamp = offset + (i * interval)
        frame_number = int(timestamp * fps)
        
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Failed to read frame for {item_name} at {timestamp}s")
            continue
        
        # Extract ROI
        roi_img = frame[y1:y2, x1:x2]
        
        # Preprocess
        preprocessed = preprocess_roi(roi_img)
        
        # Save
        output_path = os.path.join(output_dir, f"{filename}.png")
        cv2.imwrite(output_path, preprocessed)
        seen_items.add(filename)
        
        print(f"Saved: {output_path}")
    
    cap.release()

def extract_character_names_and_costumes(video_path, characters_list):
    """Extract character names and costumes (if they exist)"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output directories
    os.makedirs('images/characters', exist_ok=True)
    os.makedirs('images/costumes', exist_ok=True)
    
    # Track unique names
    seen_base_names = set()
    seen_costumes = set()
    
    for i, full_name in enumerate(characters_list):
        timestamp = 0.5 + (i * 1.0)
        frame_number = int(timestamp * fps)
        
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Failed to read frame for {full_name}")
            continue
        
        # Parse character name and costume
        if '(' in full_name:
            base_name = full_name.split('(')[0].strip()
            costume = full_name.split('(')[1].rstrip(')')
            has_costume = True
        else:
            base_name = full_name
            costume = None
            has_costume = False
        
        # Extract and save CHARACTER NAME (skip duplicates)
        base_filename = sanitize_filename(base_name)
        
        if base_filename not in seen_base_names:
            x1, y1, x2, y2 = CHAR_NAME_ROI
            char_roi = frame[y1:y2, x1:x2]
            char_preprocessed = preprocess_roi(char_roi)
            
            char_output = f"images/characters/{base_filename}.png"
            cv2.imwrite(char_output, char_preprocessed)
            seen_base_names.add(base_filename)
            print(f"Saved character: {char_output}")
        else:
            print(f"Skipping duplicate character: {base_name}")
        
        # Extract and save COSTUME (skip duplicates)
        if has_costume:
            costume_filename = sanitize_filename(costume)
            
            if costume_filename not in seen_costumes:
                x1, y1, x2, y2 = COSTUME_ROI
                costume_roi = frame[y1:y2, x1:x2]
                costume_preprocessed = preprocess_roi(costume_roi)
                
                costume_output = f"images/costumes/{costume_filename}.png"
                cv2.imwrite(costume_output, costume_preprocessed)
                seen_costumes.add(costume_filename)
                print(f"Saved costume: {costume_output}")
            else:
                print(f"Skipping duplicate costume: {costume}")
    
    cap.release()

# Create screen indicators directory
os.makedirs('images/screens', exist_ok=True)

# Extract screen indicators
print("Extracting screen indicators...")
extract_screen_indicator('D:/Resolve/Export/CharMetronome.mp4', 'images/screens/character_screen.png')
extract_screen_indicator('D:/Resolve/Export/KartMetronome.mp4', 'images/screens/kart_screen.png')

# Run extraction
print("\nExtracting characters and costumes...")
extract_character_names_and_costumes('D:/Resolve/Export/CharMetronome.mp4', KNOWN_CHARACTERS)

print("\nExtracting karts...")
extract_frames_at_intervals(
    'D:/Resolve/Export/KartMetronome.mp4',
    KNOWN_KARTS,
    KART_NAME_ROI,
    'images/karts',
    interval=1.0,
    offset=0.5
)

print("\nDone! Check the images/ folder for results.")