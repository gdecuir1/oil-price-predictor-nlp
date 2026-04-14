from os import dup
from typing import Set

duplicate: int = 0  # Stores the number of duplicate elements found
unique: Set[str] = set()  # Set to store all the unique files
with open("current_output.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line in unique:
            duplicate += 1
        else:
            unique.add(line)


print(f"There are {duplicate} elements found in the the file")
