import os
from src.preprocessing.face_extraction import FaceExtractor

extractor = FaceExtractor()
base_path = "C:/Users/rashi/deepfake detection/data"
val_real_dir = os.path.join(base_path, "processed_faces/val/real")
val_fake_dir = os.path.join(base_path, "processed_faces/val/fake")

os.makedirs(val_real_dir, exist_ok=True)
os.makedirs(val_fake_dir, exist_ok=True)

# Extract from one real video
real_video = os.path.join(base_path, "original_sequences/youtube/c23/videos/437.mp4")
extractor.extract_faces(real_video, val_real_dir, frame_rate=0.5)

# Extract from one fake video
fake_video = os.path.join(base_path, "manipulated_sequences/Deepfakes/c23/videos/033_097.mp4")
extractor.extract_faces(fake_video, val_fake_dir, frame_rate=0.5)

print("Restored some images to val/real and val/fake")
