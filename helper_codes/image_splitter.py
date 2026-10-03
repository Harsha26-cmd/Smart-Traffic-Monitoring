import os
import shutil
import math
import random

SOURCE_DIR = "dataset/images"
OUTPUT_DIR = "labeling_batches"
NUM_FOLDERS = 10

def distribute_images():
    all_images = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith((".jpg",".jpeg",".png",".bmp")):
                all_images.append(os.path.join(root,file))

    total_images = len(all_images)
    if total_images == 0:
        print("No images found!")
        return

    random.shuffle(all_images)
    images_per_folder = math.ceil(total_images / NUM_FOLDERS)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i in range(NUM_FOLDERS):
        folder_path = os.path.join(OUTPUT_DIR, f"batch_{i+1}")
        os.makedirs(folder_path, exist_ok=True)
        for img_path in all_images[i*images_per_folder:min((i+1)*images_per_folder,total_images)]:
            try:
                shutil.copy(img_path, folder_path)
            except Exception as e:
                print(f"Error copying {img_path}: {e}")

if __name__ == "__main__":
    distribute_images()
