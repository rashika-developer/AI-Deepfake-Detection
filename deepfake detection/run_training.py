from src.train import train_model
import os

if __name__ == "__main__":
    data_dir = "C:/Users/rashi/deepfake detection/data/processed_faces"
    
    # Check if we have data
    train_real = os.listdir(os.path.join(data_dir, "train", "real"))
    train_fake = os.listdir(os.path.join(data_dir, "train", "fake"))
    
    print(f"Found {len(train_real)} real and {len(train_fake)} fake images for training.")
    
    if len(train_real) > 0 and len(train_fake) > 0:
        # Train for 10 epochs with the expanded artifact set
        train_model(data_dir, model_name='efficientnet_b0', num_epochs=10, batch_size=16, lr=0.0001)
    else:
        print("Error: No data found for training.")
