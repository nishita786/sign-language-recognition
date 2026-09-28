from pathlib import Path

DATASET_PATH = Path("dataset/raw")

video_extensions = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm"
}

videos = [
    file
    for file in DATASET_PATH.rglob("*")
    if file.suffix.lower() in video_extensions
]

print("=" * 50)
print("ISL DATASET INSPECTION")
print("=" * 50)

print("\nDataset location:")
print(DATASET_PATH.resolve())

print(f"\nTotal videos found: {len(videos)}")

print("\nFirst 20 videos:")

for video in videos[:20]:
    print(video)

print("\n" + "=" * 50)