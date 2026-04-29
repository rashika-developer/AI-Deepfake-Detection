from src.inference import VideoInference
import sys
import os
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python detect.py <path_to_video_or_image>")
        return

    input_path = sys.argv[1]
    
    # Model paths
    eff_path = "src/models/best_deepfake_detector_efficientnet_b0.pth"
    vit_path = "src/models/best_deepfake_detector_vit.pth"
    
    # Fallback for original naming
    if not os.path.exists(eff_path) and os.path.exists("src/models/best_deepfake_detector.pth"):
        eff_path = "src/models/best_deepfake_detector.pth"

    print(f"\nAnalyzing: {input_path}")
    print("-" * 50)
    
    detectors = []
    if os.path.exists(eff_path):
        detectors.append(('EfficientNet', VideoInference(eff_path, model_name='efficientnet_b0')))
    if os.path.exists(vit_path):
        detectors.append(('ViT', VideoInference(vit_path, model_name='vit')))

    if not detectors:
        print("Error: No model weights found.")
        return

    # Check if input is image or video
    ext = os.path.splitext(input_path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png']:
        import cv2
        img = cv2.imread(input_path)
        if img is None:
            print("Error: Could not read image.")
            return
            
        # Face Detection
        from src.preprocessing.face_extraction import FaceExtractor
        extractor = FaceExtractor()
        faces = extractor.detect_faces(img)
        
        if len(faces) == 0:
            print("No face detected by cascade. Attempting direct inference on full image...")
            probs = []
            for name, d in detectors:
                prob = d.predict_image(img)
                label = "FAKE" if prob > 0.5 else "REAL"
                conf = prob if prob > 0.5 else 1 - prob
                print(f"- {name} Score: {label} ({conf:.2%})")
                probs.append(prob)
            
            avg_prob = np.mean(probs)
            final_label = "FAKE" if avg_prob > 0.5 else "REAL"
            final_conf = avg_prob if avg_prob > 0.5 else 1 - avg_prob
            print(f"\nFinal Detection Result: {final_label}")
            print(f"Ensemble Confidence: {final_conf:.2%}")
            return
            
        for i, (x, y, w, h) in enumerate(faces):
            print(f"\nProcessing Face {i+1}:")
            face = img[y:y+h, x:x+w]
            probs = []
            for name, d in detectors:
                prob = d.predict_image(face)
                label = "FAKE" if prob > 0.5 else "REAL"
                conf = prob if prob > 0.5 else 1 - prob
                print(f"- {name} Score: {label} ({conf:.2%})")
                probs.append(prob)
            
            avg_prob = np.mean(probs)
            final_label = "FAKE" if avg_prob > 0.5 else "REAL"
            final_conf = avg_prob if avg_prob > 0.5 else 1 - avg_prob
            print(f"Face Result: {final_label} ({final_conf:.2%}) [Ensemble Consensus]")
            
    else:
        # Video Detection
        all_probs = []
        individual_model_probs = {}

        for name, d in detectors:
            label, conf = d.predict_video(input_path)
            if label:
                # Convert confidence back to raw probability (0.0 to 1.0)
                prob = conf if label == "FAKE" else 1 - conf
                print(f"- {name} Video Result: {label} ({conf:.2%})")
                all_probs.append(prob)
        
        if not all_probs:
            print("No faces detected in video.")
            return
            
        avg_prob = np.mean(all_probs)
        final_label = "FAKE" if avg_prob > 0.5 else "REAL"
        final_conf = avg_prob if avg_prob > 0.5 else 1 - avg_prob
        print(f"\nFinal Detection Result: {final_label}")
        print(f"Ensemble Confidence: {final_conf:.2%}")

if __name__ == "__main__":
    main()
