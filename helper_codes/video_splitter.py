import cv2
import os
import random

DATASET_DIR = "dataset"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
INTERVAL_SECONDS = 1

def create_directory_structure(base_dir):
    subdirs = ["train/images","train/labels","test/images","test/labels","val/images","val/labels"]
    os.makedirs(base_dir, exist_ok=True)
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

def split_video(video_path, video_name, dataset_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, interval_seconds=1):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    saved_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Could not retrieve FPS.")
        return

    frame_interval = max(1, int(fps * interval_seconds))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            rn = random.random()
            split = "train" if rn < train_ratio else ("val" if rn < train_ratio + val_ratio else "test")
            image_name = f"{video_name}_frame_{saved_count:06d}.jpg"
            cv2.imwrite(os.path.join(dataset_dir, split, "images", image_name), frame)
            label_name = f"{video_name}_frame_{saved_count:06d}.txt"
            open(os.path.join(dataset_dir, split, "labels", label_name), "w").close()
            saved_count += 1
        frame_count += 1

    cap.release()

if __name__ == "__main__":
    create_directory_structure(DATASET_DIR)
    VIDEO_DIR = "Videos"
    if not os.path.exists(VIDEO_DIR):
        print(f"Error: video directory '{VIDEO_DIR}' not found.")
        raise SystemExit(1)

    target_video = "Final CCTV Rec.mp4"
    video_files = [f for f in os.listdir(VIDEO_DIR) if f == target_video]
    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        split_video(video_path, os.path.splitext(video_file)[0], DATASET_DIR,
                    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, INTERVAL_SECONDS)
