def reorder_array(arr):
    # Mapping of where each position should go
    position_map = {
        0: 2,
        1: 3,
        2: 8,
        3: 9,
        4: 1,
        5: 4,
        6: 7,
        7: 10,
        8: 0,
        9: 5,
        10: 6,
        11: 11
    }
    
    # The first 12 elements are already correctly ordered
    result = arr[:12].copy()
    
    # Process elements from index 12 onwards
    if len(arr) > 12:
        # Create a temporary array for the jumbled elements
        jumbled = arr[12:]
        fixed = [None] * len(jumbled)
        
        for i in range(len(jumbled)):
            # Determine which pattern position this is (0-11)
            pattern_pos = i % 12
            # Get where this should go in the pattern
            target_offset = position_map[pattern_pos]
            # Calculate the actual target index
            target_index = (i // 12) * 12 + target_offset
            
            # Place the element at the correct position
            if target_index < len(fixed):
                fixed[target_index] = jumbled[i]
        
        result.extend(fixed)
    
    return result

# Example usage with the Mario characters
characters = [
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

reordered = reorder_array(characters)

# Print as a Python array
print("[")
for i, char in enumerate(reordered):
    if i < len(reordered) - 1:
        print(f'    "{char}",')
    else:
        print(f'    "{char}"')
print("]")