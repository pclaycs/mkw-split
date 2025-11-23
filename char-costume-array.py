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

def parse_character_costumes(character_list):
    """Parse character list to extract base characters and their costumes"""
    
    # Dictionary to store character -> list of costumes
    character_costumes = {}
    
    # Set to track unique base character names
    base_characters = set()
    
    for char in character_list:
        if '(' in char and ')' in char:
            # Has a costume
            base_name = char.split('(')[0].strip()
            costume = char.split('(')[1].split(')')[0].strip()
            
            base_characters.add(base_name)
            
            if base_name not in character_costumes:
                character_costumes[base_name] = []
            
            character_costumes[base_name].append(costume)
        else:
            # No costume - just base character
            base_characters.add(char)
            if char not in character_costumes:
                character_costumes[char] = []
    
    return character_costumes, sorted(base_characters)

def get_unique_costumes(character_costumes):
    """Extract all unique costume types"""
    all_costumes = set()
    for costumes in character_costumes.values():
        all_costumes.update(costumes)
    return sorted(all_costumes)

# Parse the data
character_costumes, unique_characters = parse_character_costumes(KNOWN_CHARACTERS)
unique_costumes = get_unique_costumes(character_costumes)

# Print the results
print("# Unique base character names:")
print(f"UNIQUE_CHARACTERS = {unique_characters}")
print(f"\n# Total unique characters: {len(unique_characters)}")

print("\n# All unique costume types:")
print(f"UNIQUE_COSTUMES = {unique_costumes}")
print(f"\n# Total unique costumes: {len(unique_costumes)}")

print("\n# Character -> Costumes mapping:")
print("CHARACTER_COSTUMES = {")
for char, costumes in sorted(character_costumes.items()):
    if costumes:
        print(f'    "{char}": {costumes},')
    else:
        print(f'    "{char}": [],')
print("}")

print(f"\n# Characters with costumes: {sum(1 for c in character_costumes.values() if c)}")
print(f"# Characters without costumes: {sum(1 for c in character_costumes.values() if not c)}")