# Deepfake Detection System

A deep learning-based system for detecting deepfake images and videos using multiple model architectures (EfficientNet-B0, ResNet50, ViT). This project provides end-to-end functionality for data preprocessing, model training, and inference.

## Features

- **Multi-Model Support**: EfficientNet-B0, ResNet50, and Vision Transformer (ViT)
- **Face Extraction**: Automatic face detection and alignment from videos/images using MediaPipe
- **Video Analysis**: Frame-by-frame analysis with ensemble predictions
- **High Accuracy**: Pre-trained weights for immediate inference
- **Flexible Training**: Configurable batch sizes, learning rates, and epochs

## Project Structure

```
.
├── src/                          # Core library
│   ├── models/
│   │   └── detector.py          # DeepfakeDetector model architecture
│   ├── preprocessing/
│   │   └── face_extraction.py   # Face detection and extraction
│   ├── utils/
│   │   └── dataset.py           # Dataset loading utilities
│   ├── train.py                 # Training pipeline
│   └── inference.py             # Inference and detection
├── data/                         # Data directory
│   ├── processed_faces/
│   │   ├── train/
│   │   │   ├── real/
│   │   │   └── fake/
│   │   └── val/
│   │       ├── real/
│   │       └── fake/
│   └── original_sequences/       # Raw dataset storage
├── run_training.py              # Quick training script
├── detect.py                    # CLI detection tool
├── quick_setup.py               # Quick data setup
├── setup_data.py                # Full data preprocessing
├── train_vit.py                 # Vision Transformer training
├── analyze_performance.py       # Model performance analysis
├── dataset downloader.py        # Dataset download utility
└── extract_test_frames.py       # Test frame extraction
```

## Installation

1. **Clone the repository** (if applicable)

2. **Install dependencies**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install opencv-python mediapipe numpy pillow tqdm
   ```

## Quick Start

### 1. Setup Data

Process your dataset to extract faces:

```bash
python quick_setup.py
```

For full data setup from raw videos:
```bash
python setup_data.py
```

### 2. Train Model

Train with EfficientNet-B0 (default):
```bash
python run_training.py
```

Train with Vision Transformer:
```bash
python train_vit.py
```

Custom training:
```python
from src.train import train_model

train_model(
    data_dir='data/processed_faces',
    model_name='efficientnet_b0',  # or 'resnet50', 'vit'
    num_epochs=10,
    batch_size=16,
    lr=0.0001
)
```

### 3. Run Detection

**Command line usage**:
```bash
python detect.py <path_to_video_or_image>
```

**Python API**:
```python
from src.inference import VideoInference

# Load model
detector = VideoInference('src/models/best_deepfake_detector_efficientnet_b0.pth', 
                          model_name='efficientnet_b0')

# Predict on video
probability = detector.predict_video('test_video.mp4', frame_rate=2)
print(f"Deepfake probability: {probability:.2%}")

# Or predict on image
import cv2
img = cv2.imread('test_image.jpg')
face = detector.face_extractor.extract_face(img)
if face is not None:
    prob = detector.predict_image(face)
    print(f"Deepfake probability: {prob:.2%}")
```

## Data Structure

Organize your processed data as follows:

```
data/processed_faces/
├── train/
│   ├── real/        # Real face images
│   └── fake/        # Deepfake face images
└── val/
    ├── real/
    └── fake/
```

## How It Works

### Preprocessing
- **Face Detection**: MediaPipe detects facial landmarks and bounding boxes
- **Face Extraction**: Crops faces with 20% margin to capture peripheral artifacts
- **Normalization**: Resizes to 224×224 and applies ImageNet normalization

### Model Architecture
Supports three architectures via the `DeepfakeDetector` class:
- **EfficientNet-B0**: Optimal balance of speed and accuracy
- **ResNet50**: Strong feature extraction
- **Vision Transformer (ViT-B-16)**: State-of-the-art accuracy

Each model uses:
- Sigmoid activation for binary classification
- Dropout (0.5) for regularization
- Binary cross-entropy loss for training

### Inference
- Extracts faces from each frame of the video
- Computes prediction probability for each face
- Returns the mean probability across all frames
- Classifies as "Deepfake" if probability > 0.5

## Model Performance

Run performance analysis:
```bash
python analyze_performance.py
```

## Troubleshooting

**No model weights found**
- Ensure model files exist in `src/models/`
- Check file naming: `best_deepfake_detector_{model_name}.pth`

**CUDA out of memory**
- Reduce batch size: `batch_size=8`
- Use CPU: Set `device='cpu'` in inference

**No faces detected**
- Ensure faces are visible and at reasonable resolution
- Check video quality and lighting

## Testing

Test videos and images are included for validation:
- `web_test_real_video.mp4` - Sample real video
- `web_test_fake_video.mp4` - Sample deepfake video
- `web_test_face.jpg` - Sample face image

## Requirements

- Python 3.8+
- PyTorch with CUDA support (recommended)
- OpenCV
- MediaPipe
- NumPy, Pillow, tqdm

## License

See LICENSE file for details.
