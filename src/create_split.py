from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd

DATASET_PATH = Path("dataset/raw/sign project")
OUTPUT_PATH = Path("dataset/splits")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

videos = []
labels = []

for class_dir in sorted(DATASET_PATH.iterdir()):
    if not class_dir.is_dir():
        continue

    label = class_dir.name

    for video in class_dir.glob("*.mp4"):
        videos.append(str(video))
        labels.append(label)

df = pd.DataFrame({
    "video": videos,
    "label": labels
})

print("=" * 60)
print("DATASET SPLIT")
print("=" * 60)

print(f"\nTotal videos: {len(df)}")
print(f"Total classes: {df['label'].nunique()}")

# 70% train, 30% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

# Split remaining 30% equally
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

train_df.to_csv(OUTPUT_PATH / "train.csv", index=False)
val_df.to_csv(OUTPUT_PATH / "val.csv", index=False)
test_df.to_csv(OUTPUT_PATH / "test.csv", index=False)

print("\nSplit:")
print(f"Train      : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Test       : {len(test_df)}")

print("\nClasses:")
print(sorted(df["label"].unique()))

print("\nSaved:")
print(OUTPUT_PATH / "train.csv")
print(OUTPUT_PATH / "val.csv")
print(OUTPUT_PATH / "test.csv")