import cv2
from pathlib import Path
from collections import Counter

DATASET_PATH = Path("dataset/raw/sign project")

videos = list(DATASET_PATH.rglob("*.mp4"))

print("=" * 60)
print("ISL VIDEO DATASET INSPECTION")
print("=" * 60)

print(f"\nTotal videos: {len(videos)}")

resolutions = Counter()
fps_values = Counter()
failed = []

for i, video_path in enumerate(videos, 1):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        failed.append(str(video_path))
        continue

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    resolutions[(width, height)] += 1
    fps_values[round(fps, 2)] += 1

    cap.release()

    if i % 100 == 0:
        print(f"Checked {i}/{len(videos)} videos...")

print("\n" + "=" * 60)
print("RESOLUTION")
print("=" * 60)

for resolution, count in resolutions.most_common():
    print(f"{resolution}: {count} videos")

print("\n" + "=" * 60)
print("FPS")
print("=" * 60)

for fps, count in fps_values.most_common():
    print(f"{fps} FPS: {count} videos")

print("\n" + "=" * 60)
print("FAILED VIDEOS")
print("=" * 60)

print(f"Failed: {len(failed)}")

for video in failed[:20]:
    print(video)

print("\nInspection complete.")