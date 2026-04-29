import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Assuming directory structure:
        # root_dir/
        #   real/
        #     img1.jpg, img2.jpg...
        #   fake/
        #     img1.jpg, img2.jpg...
        
        for label, class_dir in enumerate(['real', 'fake']):
            dir_path = os.path.join(root_dir, class_dir)
            if not os.path.exists(dir_path):
                continue
            for img_name in os.listdir(dir_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(dir_path, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloader(root_dir, batch_size=32, shuffle=True, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = DeepfakeDataset(root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

if __name__ == "__main__":
    # Example usage
    # dataloader = get_dataloader('data/processed_faces')
    # for images, labels in dataloader:
    #     print(images.shape, labels.shape)
    #     break
    pass
