import torch
import cv2
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from src.models.detector import DeepfakeDetector
from src.preprocessing.face_extraction import FaceExtractor

class VideoInference:
    def __init__(self, model_path, model_name='efficientnet_b0', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = DeepfakeDetector(model_name=model_name, pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.face_extractor = FaceExtractor(margin=0.2)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict_image(self, face_img):
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL and Transform
        face_pil = Image.fromarray(face_img)
        face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(face_tensor)
            probability = output.item()
            return probability

    def predict_video(self, video_path, frame_rate=2):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(fps / frame_rate) if frame_rate > 0 else 1
        
        frame_count = 0
        probabilities = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % interval == 0:
                # Use Haar Cascade for detection
                faces = self.face_extractor.detect_faces(frame)
                
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        ih, iw, _ = frame.shape
                        
                        # Add margin
                        x_margin, y_margin = int(w * 0.2), int(h * 0.2)
                        x1, y1 = max(0, x - x_margin), max(0, y - y_margin)
                        x2, y2 = min(iw, x + w + x_margin), min(ih, y + h + y_margin)
                        
                        face = frame[y1:y2, x1:x2]
                        if face.size > 0:
                            prob = self.predict_image(face)
                            probabilities.append(prob)
            
            frame_count += 1
            
        cap.release()
        
        if not probabilities:
            return None, "No face detected"
        
        avg_prob = np.mean(probabilities)
        label = "FAKE" if avg_prob > 0.5 else "REAL"
        confidence = avg_prob if avg_prob > 0.5 else 1 - avg_prob
        
        return label, confidence

if __name__ == "__main__":
    # Example usage:
    # detector = VideoInference('src/models/best_deepfake_detector.pth')
    # label, conf = detector.predict_video('test_video.mp4')
    # print(f"Result: {label} with {conf:.2f} confidence")
    pass
