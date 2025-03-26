import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Get classes
        split_dir = os.path.join(root_dir, split)
        self.classes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        self.classes.sort()  # Sort to ensure consistent class indices
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Get all image paths and labels
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(split_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                # if img_name.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp')):
                img_path = os.path.join(class_dir, img_name)
                self.images.append(img_path)
                self.labels.append(class_idx)
        
        print(f"Loaded {len(self.images)} images for {split} split across {len(self.classes)} classes")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        # Handle potential file errors gracefully
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image and the same label
            placeholder = torch.zeros(3, 512, 512) if self.transform is None else self.transform(Image.new('RGB', (512, 512)))
            return placeholder, self.labels[idx]


# Example usage:
if __name__ == "__main__":
    # Define transforms for the training and validation/test data with 512x512 dimensions
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(512),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(576),  # Resize the smallest side to 576
        transforms.CenterCrop(512),  # Then center crop to 512x512
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ImageDataset(root_dir='../data', split='train', transform=train_transform)
    val_dataset = ImageDataset(root_dir='../data', split='val', transform=val_transform)
    test_dataset = ImageDataset(root_dir='../data', split='test', transform=val_transform)
    
    # Create data loaders with appropriate batch size for large images
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=8,  # Smaller batch size due to larger images
        shuffle=True, 
        num_workers=4,
        pin_memory=True  # Helps speed up data transfer to GPU
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=8, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=8, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Check image dimensions
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        break