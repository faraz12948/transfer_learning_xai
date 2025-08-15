import os
import shutil
import pandas as pd

# ======== CONFIG ========
csv_path = "./dataset/NewDataset/CSV_datafiles/_valid_classes.csv"  # Path to your CSV file
source_folder = "./dataset/NewDataset/valid"  # Folder containing the files
destination_base = "./dataset/NewDataset"  # Base folder where sorted images will be moved

# Mapping from CSV column name to folder name
folder_map = {
    "MD": "MildDemented",
    "ND": "NonDemented",
    "VMD": "VeryMildDemented",
    "MoD": "ModerateDemented"
}

# ======== SCRIPT ========
# Read CSV
df = pd.read_csv(csv_path)

# Ensure destination folders exist
for folder in folder_map.values():
    os.makedirs(os.path.join(destination_base, folder), exist_ok=True)

# Iterate over rows
for _, row in df.iterrows():
    filename = row["filename"].strip()
    src_path = os.path.join(source_folder, filename)

    if not os.path.exists(src_path):
        print(f"⚠ File not found: {src_path}")
        continue

    # Find the column with value 1
    moved = False
    for col, dest_folder in folder_map.items():
        print(f"Checking {filename} for label {col}...")
        if row[col] == 1:
            dest_path = os.path.join(destination_base, dest_folder, filename)
            shutil.move(src_path, dest_path)
            print(f"Moved: {filename} → {dest_folder}")
            moved = True
            break

    if not moved:
        print(f"⚠ No matching label found for {filename}")

print("✅ Done!")
