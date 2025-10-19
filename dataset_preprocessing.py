import os
import shutil
import random
from pathlib import Path
import hashlib

base_dir = Path("./dataset")  
output_dir = Path("./split_dataset")
folders = ["good", "bad", "unsafe to drive", "change soon"]
split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
random.seed(42)


for split in split_ratios.keys():
    for folder in folders:
        (output_dir / split / folder).mkdir(parents=True, exist_ok=True)


for folder in folders:
    images = list((base_dir / folder).glob("*"))
    random.shuffle(images)
    n = len(images)
    n_train = int(n * split_ratios["train"])
    n_val = int(n * split_ratios["val"])

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    for img_path in train_imgs:
        shutil.copy(img_path, output_dir / "train" / folder / img_path.name)
    for img_path in val_imgs:
        shutil.copy(img_path, output_dir / "val" / folder / img_path.name)
    for img_path in test_imgs:
        shutil.copy(img_path, output_dir / "test" / folder / img_path.name)

print(" Dataset splitted")


def file_hash(filepath):
    """Return md5 hash of a file."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def collect_hashes(folder_path):
    hash_dict = {}
    for file in Path(folder_path).rglob("*"):
        if file.is_file():
            h = file_hash(file)
            hash_dict[h] = file
    return hash_dict

# Collect hashes
train_hashes = collect_hashes(output_dir / "train")
test_hashes = collect_hashes(output_dir / "test")
val_hashes = collect_hashes(output_dir / "val")

# Detect duplicates
duplicates_train_test = set(train_hashes.keys()) & set(test_hashes.keys())
duplicates_train_val = set(train_hashes.keys()) & set(val_hashes.keys())
duplicates_val_test = set(val_hashes.keys()) & set(test_hashes.keys())

print(f"Duplicates (train-test): {len(duplicates_train_test)}")
print(f"Duplicates (train-val): {len(duplicates_train_val)}")
print(f"Duplicates (val-test): {len(duplicates_val_test)}")

# Remove duplicates between train and test
for dup_hash in duplicates_train_test:
    file_to_remove = test_hashes[dup_hash]
    os.remove(file_to_remove)
    print(f"UPDATED: {file_to_remove}")
