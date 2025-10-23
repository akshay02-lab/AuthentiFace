import os, shutil, random

# paths
src_folder = "C:/Users/aksha/OneDrive/Desktop/Liveness_Detection/dataset"  # folder with mixed images
dst_folder = "C:/Users/aksha/OneDrive/Desktop/Liveness_Detection/organized"
    # new root folder for train/val/test

# make splits
splits = ["train_inc", "val_inc", "test_inc"]
classes = ["real", "fake"]

for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(dst_folder, split, cls), exist_ok=True)

# collect all files
all_files = [f for f in os.listdir(src_folder) if f.endswith((".jpg", ".png"))]

# shuffle to randomize
random.shuffle(all_files)

# split 70% train, 15% val, 15% test
n = len(all_files)
train_files = all_files[:int(0.7*n)]
val_files   = all_files[int(0.7*n):int(0.85*n)]
test_files  = all_files[int(0.85*n):]

def move_files(file_list, split):
    for fname in file_list:
        if "real" in fname.lower():
            cls = "real"
        elif "fake" in fname.lower():
            cls = "fake"
        else:
            continue
        src = os.path.join(src_folder, fname)
        dst = os.path.join(dst_folder, split, cls, fname)
        shutil.copy(src, dst)

# move files into organized structure
move_files(train_files, "train_inc")
move_files(val_files, "val_inc")
move_files(test_files, "test_inc")

print("✅ Files organized into train/val/test with real/fake subfolders!")
