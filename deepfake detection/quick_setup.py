import os
from src.preprocessing.face_extraction import FaceExtractor
import shutil
import random

def quick_setup():
    extractor = FaceExtractor()
    base_data_path = "C:/Users/rashi/deepfake detection/data"
    processed_path = os.path.join(base_data_path, "processed_faces")
    
    # Target directories
    for split in ['train', 'val']:
        for label in ['real', 'fake']:
            os.makedirs(os.path.join(processed_path, split, label), exist_ok=True)
    
    real_video_dir = os.path.join(base_data_path, "original_sequences/youtube/c23/videos")
    fake_sources = [
        os.path.join(base_data_path, "manipulated_sequences/Deepfakes/c23/videos"),
        os.path.join(base_data_path, "manipulated_sequences/FaceSwap/c23/videos")
    ]
    
    # Quick process: 10 real, 10 fake
    real_videos = [v for v in os.listdir(real_video_dir) if v.endswith(".mp4")][:10]
    for v in real_videos:
        extractor.extract_faces(os.path.join(real_video_dir, v), os.path.join(processed_path, "train", "real"), frame_rate=1)
        
    for source in fake_sources:
        fake_videos = [v for v in os.listdir(source) if v.endswith(".mp4")][:5]
        for v in fake_videos:
            extractor.extract_faces(os.path.join(source, v), os.path.join(processed_path, "train", "fake"), frame_rate=1)

    print("Quick data setup complete.")

if __name__ == "__main__":
    quick_setup()
