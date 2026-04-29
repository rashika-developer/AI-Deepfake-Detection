import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader
from torchvision import transforms
from src.models.detector import DeepfakeDetector
from src.utils.dataset import DeepfakeDataset
from tqdm import tqdm

def evaluate_model(model_name, model_path, val_loader, device):
    model = DeepfakeDetector(model_name=model_name, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    y_true = []
    y_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Evaluating {model_name}"):
            images = images.to(device)
            outputs = model(images)
            y_probs.extend(outputs.cpu().numpy().flatten())
            y_true.extend(labels.numpy().flatten())
            
    return np.array(y_true), np.array(y_probs)

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = "C:/Users/rashi/deepfake detection/data/processed_faces"
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_dataset = DeepfakeDataset(os.path.join(data_dir, 'val'), transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Models to evaluate
    models_to_test = [
        ('efficientnet_b0', 'src/models/best_deepfake_detector_efficientnet_b0.pth'),
        ('vit', 'src/models/best_deepfake_detector_vit.pth')
    ]
    
    results = {}
    
    for name, path in models_to_test:
        if os.path.exists(path):
            y_true, y_probs = evaluate_model(name, path, val_loader, device)
            results[name] = (y_true, y_probs)
        else:
            print(f"Skipping {name}, path not found: {path}")

    if not results:
        print("No results to analyze.")
        return

    # Ensemble logic
    if len(results) == 2:
        y_true = results['efficientnet_b0'][0]
        ensemble_probs = (results['efficientnet_b0'][1] + results['vit'][1]) / 2
        results['ensemble'] = (y_true, ensemble_probs)

    # Generate Metrics
    for name, (y_true, y_probs) in results.items():
        y_pred = (y_probs > 0.5).astype(int)
        print(f"\n--- {name.upper()} Analysis ---")
        print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))
        
        plot_confusion_matrix(y_true, y_pred, f'Confusion Matrix: {name}', f'analysis_{name}_cm.png')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend()
    plt.savefig('analysis_roc_curve.png')
    print("\nVisualizations saved: analysis_efficientnet_b0_cm.png, analysis_vit_cm.png, analysis_ensemble_cm.png, analysis_roc_curve.png")

if __name__ == "__main__":
    main()
