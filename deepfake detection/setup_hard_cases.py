import os
from src.preprocessing.face_extraction import FaceExtractor

def setup_hard_cases():
    extractor = FaceExtractor()
    processed_path = "C:/Users/rashi/deepfake detection/data/processed_faces"
    
    # Target directories for "hard" training
    train_fake_dir = os.path.join(processed_path, "train", "fake")
    os.makedirs(train_fake_dir, exist_ok=True)
    
    # Specifically target the videos that failed
    failed_videos = [
        "C:/Users/rashi/deepfake detection/data/manipulated_sequences/Deepfakes/c23/videos/hard_fake_1.mp4",
        "C:/Users/rashi/deepfake detection/data/manipulated_sequences/Deepfakes/c23/videos/hard_fake_2.mp4"
    ]
    
    for v in failed_videos:
        print(f"Extracting hard samples from {v}...")
        # Higher frame rate for these specifically
        extractor.extract_faces(v, train_fake_dir, frame_rate=10)

    print("Hard cases added to training set.")

if __name__ == "__main__":
    setup_hard_cases()
