import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------------------------
# Find one processed video
# ------------------------------------------------------------

processed_dir = Path("dataset/processed/train")

files = sorted(processed_dir.glob("*.npy"))

file_path = files[0]

print("Showing:", file_path.name)

# Load sequence
sequence = np.load(file_path)

print("Shape:", sequence.shape)


# ------------------------------------------------------------
# Select 6 frames
# ------------------------------------------------------------

indices = [0, 6, 12, 18, 24, 31]


# ------------------------------------------------------------
# Display frames
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 8)
)

for ax, index in zip(
    axes.ravel(),
    indices
):

    ax.imshow(sequence[index])

    ax.set_title(
        f"Frame {index + 1}"
    )

    ax.axis("off")


plt.tight_layout()

plt.show()