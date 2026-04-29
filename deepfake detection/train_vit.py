from src.train import train_model
import os

if __name__ == "__main__":
    data_dir = "C:/Users/rashi/deepfake detection/data/processed_faces"
    
    # Check if we have data
    if os.path.exists(data_dir):
        print(f"Starting Vision Transformer (ViT) training...")
        # Fast update for ViT to capture new FaceSwap artifacts
        train_model(data_dir, 
                    model_name='vit', 
                    num_epochs=1, 
                    batch_size=8, 
                    lr=0.0001)
    else:
        print("Error: No data found for training.")
