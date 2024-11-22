import random

names = [  "Esti", "Devarshi", "Guru", "Ayush G", "Hitesh",
    "Amol", "Shalmali", "Rahul", "Alquama", "Kinjal",
    "Nikki", "Trishna", "Shraddha Shelatkar", "Rohan", "Vibhuti",
    "Rohit. K", "Malini", "Manish", "Vikram", "Biraj"
    ]
    
random.shuffle(names)
groups = [names[i:i + 5] for i in range(0, len(names), 5)]
for i, group in enumerate(groups, 1):
    print(f"Group {i}: {', '.join(group)}")
