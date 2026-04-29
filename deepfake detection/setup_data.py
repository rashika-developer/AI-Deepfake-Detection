import os
from src.preprocessing.face_extraction import FaceExtractor
import shutil
import random

def setup_data():
    extractor = FaceExtractor()
    base_data_path = "C:/Users/rashi/deepfake detection/data"
    processed_path = os.path.join(base_data_path, "processed_faces")
    
    # Clear existing processed data
    if os.path.exists(processed_path):
        print("Clearing old processed data...")
        shutil.rmtree(processed_path)
    
    # Paths
    real_video_dir = os.path.join(base_data_path, "original_sequences/youtube/c23/videos")
    # Multiple sources for FAKE to cover different artifacts
    fake_sources = [
        os.path.join(base_data_path, "manipulated_sequences/Deepfakes/c23/videos"),
        os.path.join(base_data_path, "manipulated_sequences/FaceSwap/c23/videos"),
        os.path.join(base_data_path, "manipulated_sequences/FaceShifter/c23/videos")
    ]
    
    # Target directories
    for split in ['train', 'val']:
        for label in ['real', 'fake']:
            os.makedirs(os.path.join(processed_path, split, label), exist_ok=True)
    
    # Process REAL
    videos = [v for v in os.listdir(real_video_dir) if v.endswith(".mp4")]
    random.shuffle(videos)
    # Increase to 100 real videos
    videos = videos[:100]
    split_idx = int(len(videos) * 0.8)
    for video in videos[:split_idx]:
        extractor.extract_faces(os.path.join(real_video_dir, video), os.path.join(processed_path, "train", "real"), frame_rate=4)
    for video in videos[split_idx:]:
        extractor.extract_faces(os.path.join(real_video_dir, video), os.path.join(processed_path, "val", "real"), frame_rate=4)

    # Process FAKE from multiple sources
    total_fake_target = 100
    per_source_target = total_fake_target // len(fake_sources)
    for source_dir in fake_sources:
        if not os.path.exists(source_dir): continue
        print(f"Processing fakes from {os.path.basename(os.path.dirname(os.path.dirname(source_dir)))}...")
        videos = [v for v in os.listdir(source_dir) if v.endswith(".mp4")]
        random.shuffle(videos)
        videos = videos[:per_source_target]
        split_idx = int(len(videos) * 0.8)
        for video in videos[:split_idx]:
            extractor.extract_faces(os.path.join(source_dir, video), os.path.join(processed_path, "train", "fake"), frame_rate=4)
        for video in videos[split_idx:]:
            extractor.extract_faces(os.path.join(source_dir, video), os.path.join(processed_path, "val", "fake"), frame_rate=4)

    print(f"Data setup complete at {processed_path}")
    print(f"Final Counts - Train: {len(os.listdir(os.path.join(processed_path, 'train', 'real')))}R / {len(os.listdir(os.path.join(processed_path, 'train', 'fake')))}F")

if __name__ == "__main__":
    setup_data()
