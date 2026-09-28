import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("MediaPipe version:", getattr(mp, "__version__", "unknown"))

print("\nHandLandmarker:")
print(vision.HandLandmarker)

print("\nHandLandmarkerOptions:")
print(vision.HandLandmarkerOptions)

print("\nRunningMode:")
print(vision.RunningMode)

print("\nMediaPipe Tasks API is working correctly!")
