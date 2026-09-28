import cv2
import numpy as np
from pathlib import Path

DATASET_PATH = Path("dataset/raw/sign project")

videos = list(DATASET_PATH.rglob("*.mp4"))

frame_counts = []
durations = []

for i, video_path in enumerate(videos, 1):

    cap = cv2.VideoCapture(str(video_path))

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    if fps > 0:
        duration = frames / fps
        frame_counts.append(frames)
        durations.append(duration)

    if i % 100 == 0:
        print(f"Processed {i}/{len(videos)}")

print("\n" + "=" * 60)
print("FRAME / DURATION ANALYSIS")
print("=" * 60)

print(f"\nVideos analyzed: {len(frame_counts)}")

print("\nFrames:")
print(f"Minimum : {min(frame_counts)}")
print(f"Maximum : {max(frame_counts)}")
print(f"Mean    : {np.mean(frame_counts):.1f}")
print(f"Median  : {np.median(frame_counts):.1f}")

print("\nDuration (seconds):")
print(f"Minimum : {min(durations):.2f}")
print(f"Maximum : {max(durations):.2f}")
print(f"Mean    : {np.mean(durations):.2f}")
print(f"Median  : {np.median(durations):.2f}")

print("\nPercentiles:")
for p in [10, 25, 50, 75, 90]:
    print(
        f"{p}% : "
        f"{np.percentile(frame_counts, p):.0f} frames | "
        f"{np.percentile(durations, p):.2f} sec"
    )