import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import csv
import os.path as osp

# Import from src directory structure
from src.datasets import ImageDataset  # Import your dataset class
# Import the unlabeled dataset for test prediction
from src.test_datasets import UnlabeledImageDataset, create_test_loader

def load_checkpoint(model, checkpoint_path, load_optimizer=False, optimizer=None, scheduler=None):
    """
    Load model from checkpoint
    
    Args:
        model: The model to load weights into
        checkpoint_path (str): Path to the checkpoint file
        load_optimizer: Whether to load optimizer and scheduler states
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
    
    Returns:
        model: Model with loaded weights
        dict: Additional checkpoint information (epoch, accuracy, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer and scheduler if requested
    if load_optimizer and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Extract additional information
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'accuracy': checkpoint.get('acc', 0),
        'loss': checkpoint.get('loss', 0)
    }
    
    print(f"Loaded checkpoint from epoch {info['epoch']} with accuracy {info['accuracy']:.4f}")
    
    return model, info

def create_submission(predictions, image_ids, output_file='prediction.csv'):
    """
    Create a submission file for the competition
    
    Args:
        predictions (numpy.ndarray): Predicted class indices
        image_ids (list): List of image IDs/filenames
        output_file (str): Output filename
    """
    print(f"Creating submission file: {output_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Write predictions to CSV file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'pred_label'])
        for i, pred in enumerate(predictions):
            writer.writerow([image_ids[i], int(pred)])
    
    print(f"Submission file created with {len(predictions)} predictions")

def create_test_transforms(image_size):
    """
    Create a list of test transforms for multi-scale testing
    
    Args:
        image_size (int): Base image size
    
    Returns:
        list: List of transform compositions
    """
    # Dictionary to store different transforms
    test_transforms = []
    class AspectRatioPreservingTransform:
        def __init__(self, target_size, padding_mode='reflect'):
            self.target_size = target_size
            self.padding_mode = padding_mode
            
        def __call__(self, img):
            # Get original aspect ratio
            w, h = img.size
            aspect_ratio = w / h
            
            # Resize the smaller dimension to target_size
            if w < h:
                new_w = self.target_size
                new_h = int(new_w / aspect_ratio)
            else:
                new_h = self.target_size
                new_w = int(new_h * aspect_ratio)
                
            # Resize while maintaining aspect ratio
            resized_img = transforms.Resize((new_h, new_w))(img)
            
            # Pad to make square if needed
            result = transforms.Pad(
                padding=[
                    max(0, (self.target_size - new_w) // 2),
                    max(0, (self.target_size - new_h) // 2),
                    max(0, (self.target_size - new_w + 1) // 2),
                    max(0, (self.target_size - new_h + 1) // 2)
                ],
                padding_mode=self.padding_mode
            )(resized_img)
            
            # Random crop to target size
            if new_w > self.target_size or new_h > self.target_size:
                result = transforms.RandomCrop(self.target_size)(result)
                
            return result
    
    # Basic transform (center crop) - matching the validation transform from training
    basic_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.125)),  # Resize the smallest side to 576 (if image_size=512)
        transforms.CenterCrop(image_size),           # Then center crop
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transforms.append(basic_transform)
    
    # Different scales
    scales = [0.9, 1.0, 1.1, 1.2]
    for scale in scales:
        scaled_size = int(image_size * scale)
        transform = transforms.Compose([
            transforms.Resize(int(scaled_size * 1.125)),
            transforms.CenterCrop(scaled_size),
            transforms.Resize(image_size),  # Resize back to original size
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_transforms.append(transform)
    
    # # Horizontal flip
    flip_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.125)),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(p=1.0),  # Always flip
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transforms.append(flip_transform)
    
    # # Different crops
    crop_positions = ['tl', 'tr', 'bl', 'br', 'c']  # top-left, top-right, bottom-left, bottom-right, center
    for pos in crop_positions:
        transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.125)),  # Resize to larger size for cropping
            # Custom cropping function instead of center crop
            lambda img: crop_image(img, image_size, position=pos),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_transforms.append(transform)
    
    return test_transforms

def crop_image(img, size, position):
    """
    Crop image at specific position
    
    Args:
        img: PIL Image
        size: Size to crop
        position: Position to crop (tl, tr, bl, br, c)
    
    Returns:
        PIL Image: Cropped image
    """
    width, height = img.size
    
    if position == 'tl':  # top-left
        return transforms.functional.crop(img, 0, 0, size, size)
    elif position == 'tr':  # top-right
        return transforms.functional.crop(img, 0, width - size, size, size)
    elif position == 'bl':  # bottom-left
        return transforms.functional.crop(img, height - size, 0, size, size)
    elif position == 'br':  # bottom-right
        return transforms.functional.crop(img, height - size, width - size, size, size)
    else:  # center
        return transforms.functional.center_crop(img, size)

def test_with_tta(model, data_loader, device, is_labeled=True):
    """
    Test the model with test-time augmentation on a specific transform
    
    Args:
        model: PyTorch model
        data_loader: DataLoader with specific transform
        device: Device to run inference on
        is_labeled: Whether the dataset has labels (True) or image IDs (False)
    
    Returns:
        numpy.ndarray: Softmax probabilities for each class
        list: Image IDs (only returned if is_labeled=False)
    """
    model.eval()
    all_probs = []
    all_ids = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f'Testing'):
            if is_labeled:
                inputs, _ = batch  # Labeled data returns images and labels
            else:
                inputs, ids = batch  # Unlabeled data returns images and image IDs
                all_ids.extend(ids)
                
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
    
    if is_labeled:
        return np.vstack(all_probs)
    else:
        return np.vstack(all_probs), all_ids

def main():
    parser = argparse.ArgumentParser(description='Test script with Multi-Scale TTA')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--test_dir', type=str, help='Path to test images directory (if different from data_dir)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--output_file', type=str, default='prediction.csv', help='Output file path')
    parser.add_argument('--model_id', type=str, default='resnet152', 
                        choices=['resnet152', 'resnext101_32x8d', 'resnest50', 'resnest101', 'resnest200'], help='Model type')
    parser.add_argument('--unlabeled_test', action='store_true', 
                        help='Use this for unlabeled test data (competition submission)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create test transforms
    test_transforms = create_test_transforms(args.image_size)
    
    # Choose the test directory
    test_directory = args.test_dir if args.test_dir else args.data_dir
    
    # We need to determine number of classes from the training data
    train_dataset = ImageDataset(root_dir=args.data_dir, split='train', transform=None)
    num_classes = len(train_dataset.classes)
    print(f"Found {num_classes} classes from training data")
    
    # Load model from torch hub
    print(f"Loading {args.model_id} model")
    if "resnest" in args.model_id:
        model = torch.hub.load('zhanghang1989/ResNeSt', args.model_id, pretrained=True)
    else:
        model = torch.hub.load('pytorch/vision:v0.10.0', args.model_id, pretrained=True)
    
    # Modify the final fully connected layer for the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load weights from checkpoint
    print(f"Loading weights from checkpoint: {args.checkpoint}")
    model, checkpoint_info = load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    print(f"Model loaded successfully (from epoch {checkpoint_info['epoch']})")
    
    # Determine if we're using labeled or unlabeled test data
    if args.unlabeled_test:
        # For unlabeled competition test data
        print(f"Using unlabeled test data from: {test_directory}")
        
        # For unlabeled test, we'll collect image IDs during inference
        image_ids = None
        
        # Storage for all predictions
        all_predictions = None
        
        # Run inference with each transform
        for i, transform in enumerate(test_transforms):
            print(f"Running inference with transform {i+1}/{len(test_transforms)}")
            
            # Create data loader for unlabeled test data
            test_loader, test_dataset = create_test_loader(
                test_directory, 
                transform=transform, 
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            
            # Run inference
            if i == 0:  # First run, initialize predictions and get image IDs
                predictions, image_ids = test_with_tta(model, test_loader, device, is_labeled=False)
                all_predictions = predictions
            else:  # Subsequent runs, just accumulate predictions
                predictions, _ = test_with_tta(model, test_loader, device, is_labeled=False)
                all_predictions += predictions
            
    else:
        # For labeled test data (e.g., validation)
        print(f"Using labeled test data from: {test_directory}")
        test_dataset = ImageDataset(root_dir=test_directory, split='val', transform=None)
        
        # Get image IDs from the dataset
        image_ids = [osp.basename(path) for path in test_dataset.images]
        
        print(f"Found {len(image_ids)} test images across {num_classes} classes")
        
        # Storage for all predictions
        all_predictions = np.zeros((len(test_dataset), num_classes))
        
        # Run inference with each transform
        for i, transform in enumerate(test_transforms):
            print(f"Running inference with transform {i+1}/{len(test_transforms)}")
            
            # Update dataset transform
            test_dataset.transform = transform
            
            # Create data loader
            test_loader = DataLoader(
                test_dataset, 
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
            
            # Run inference
            predictions = test_with_tta(model, test_loader, device)
            
            # Accumulate predictions
            all_predictions += predictions

    # Average predictions
    all_predictions /= len(test_transforms)
    
    # Get predicted classes
    predicted_classes = np.argmax(all_predictions, axis=1)
    # from idx to class_name
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    predicted_classes = [idx_to_class[i] for i in predicted_classes]
    # Create submission file
    create_submission(predicted_classes, image_ids, args.output_file)

if __name__ == "__main__":
    main()