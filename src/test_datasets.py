import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import os.path as osp

class UnlabeledImageDataset(Dataset):
    """
    Dataset for unlabeled test images (for submission)
    Handles flat directory structure with just image files
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the test images (or path to the test file)
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.image_ids = []
        
        # Check if root_dir is a directory or a file
        if os.path.isdir(root_dir):
            # Scan for all image files in the directory
            for img_name in os.listdir(root_dir):
                if img_name.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp')):
                    img_path = os.path.join(root_dir, img_name)
                    self.images.append(img_path)
                    # Extract image ID (filename without extension)
                    self.image_ids.append(os.path.splitext(img_name)[0])
        elif os.path.isfile(root_dir) and root_dir.endswith('.txt'):
            # If it's a text file listing image paths
            with open(root_dir, 'r') as f:
                for line in f:
                    img_path = line.strip()
                    if not os.path.exists(img_path):
                        # Try relative path
                        img_path = os.path.join(os.path.dirname(root_dir), img_path)
                    
                    if os.path.exists(img_path):
                        self.images.append(img_path)
                        # Extract image ID (filename without extension)
                        self.image_ids.append(os.path.splitext(os.path.basename(img_path))[0])
        
        print(f"Loaded {len(self.images)} unlabeled test images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image_id = self.image_ids[idx]
        
        # Handle potential file errors gracefully
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            # For unlabeled test data, we return image and image_id
            return image, image_id
        
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image
            placeholder = torch.zeros(3, 640, 640) if self.transform is None else self.transform(Image.new('RGB', (640, 640)))
            return placeholder, image_id

# Example function to create a submission-ready test loader
def create_test_loader(test_dir, transform, batch_size=8, num_workers=4):
    """
    Create a DataLoader for unlabeled test images
    
    Args:
        test_dir: Directory containing test images or text file with image paths
        transform: Transforms to apply to images
        batch_size: Batch size
        num_workers: Number of worker threads
        
    Returns:
        DataLoader for test images
    """
    dataset = UnlabeledImageDataset(test_dir, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader, dataset