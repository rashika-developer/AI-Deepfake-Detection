import cv2
import os
from src.preprocessing.face_extraction import FaceExtractor

extractor = FaceExtractor()
video_path = r"C:\Users\rashi\Downloads\469_481.mp4"
output_dir = "test_frames_fake"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(video_path)
count = 0
frame_idx = 0
while count < 5:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % 30 == 0:
        faces = extractor.detect_faces(frame)
        for i, (x, y, w, h) in enumerate(faces):
            face = frame[y:y+h, x:x+w]
            cv2.imwrite(f"{output_dir}/frame_{frame_idx}_{i}.jpg", face)
            count += 1
            if count >= 5: break
    frame_idx += 1
cap.release()
print(f"Extracted {count} faces to {output_dir}")
