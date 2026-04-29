import cv2
import os
from tqdm import tqdm
import numpy as np

class FaceExtractor:
    def __init__(self, margin=0.2):
        # Using Haar Cascades as a reliable fallback
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.margin = margin

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # More sensitive parameters for real-world angles
        return self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    def extract_faces(self, path, output_dir, frame_rate=1):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Handle single image input
        if path.lower().endswith(('.png', '.jpg', '.jpeg')):
            frame = cv2.imread(path)
            if frame is not None:
                faces = self.detect_faces(frame)
                for i, (x, y, w, h) in enumerate(faces):
                    ih, iw, _ = frame.shape
                    x_margin, y_margin = int(w * self.margin), int(h * self.margin)
                    x1, y1 = max(0, x - x_margin), max(0, y - y_margin)
                    x2, y2 = min(iw, x + w + x_margin), min(ih, y + h + y_margin)
                    face = frame[y1:y2, x1:x2]
                    if face.size > 0:
                        face_filename = f"img_{os.path.basename(path).split('.')[0]}_{i}.jpg"
                        cv2.imwrite(os.path.join(output_dir, face_filename), face)
                return len(faces)
            return 0

        # Handle video input
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 30
        interval = int(fps / frame_rate) if frame_rate > 0 else 1
        
        frame_count = 0
        saved_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % interval == 0:
                faces = self.detect_faces(frame)
                
                for (x, y, w, h) in faces:
                    ih, iw, _ = frame.shape
                    
                    # Add margin
                    x_margin = int(w * self.margin)
                    y_margin = int(h * self.margin)
                    
                    x1 = max(0, x - x_margin)
                    y1 = max(0, y - y_margin)
                    x2 = min(iw, x + w + x_margin)
                    y2 = min(ih, y + h + y_margin)
                    
                    face = frame[y1:y2, x1:x2]
                    if face.size > 0:
                        face_filename = f"{os.path.basename(path).split('.')[0]}_frame{frame_count}.jpg"
                        cv2.imwrite(os.path.join(output_dir, face_filename), face)
                        saved_count += 1
            
            frame_count += 1
            
        cap.release()
        return saved_count

if __name__ == "__main__":
    pass
