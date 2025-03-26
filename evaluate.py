import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Import from src directory
from src.datasets import ImageDataset

def load_checkpoint(model, checkpoint_path):
    """
    Load model from checkpoint
    
    Args:
        model: The model to load weights into
        checkpoint_path (str): Path to the checkpoint file
    
    Returns:
        model: Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Handle both DataParallel and regular models
    if list(checkpoint['model_state_dict'].keys())[0].startswith('module.'):
        # DataParallel models will have 'module.' prefix
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        # Non-DataParallel model
        model.load_state_dict(checkpoint['model_state_dict'])
    
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'accuracy': checkpoint.get('acc', 0),
    }
    
    print(f"Loaded checkpoint from epoch {info['epoch']} with accuracy {info['accuracy']:.4f}")
    
    return model

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

def test_with_tta(model, data_loader, device):
    """
    Test the model with test-time augmentation on a specific transform
    
    Args:
        model: PyTorch model
        data_loader: DataLoader with specific transform
        device: Device to run inference on
    
    Returns:
        numpy.ndarray: Softmax probabilities for each class
        numpy.ndarray: True labels
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            all_labels.append(labels.numpy())
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
    
    return np.vstack(all_probs), np.concatenate(all_labels)

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot and save confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the confusion matrix plot
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plt.figure(figsize=(20, 20))
    ax = sns.heatmap(cm_normalized, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    
    # Rotate x-axis labels
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()

def get_top_confused_classes(y_true, y_pred, class_names, top_n=20):
    """
    Get the most commonly confused class pairs
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        top_n: Number of top confused pairs to return
    
    Returns:
        list: Top confused class pairs with counts
    """
    cm = confusion_matrix(y_true, y_pred)
    # Zero out the diagonal (correct predictions)
    np.fill_diagonal(cm, 0)
    
    # Find indices of top confused pairs
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                # True class i was predicted as class j
                confused_pairs.append((i, j, cm[i, j]))
    
    # Sort by count (descending)
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Format result
    result = []
    for true_idx, pred_idx, count in confused_pairs[:top_n]:
        result.append({
            'true_class': class_names[true_idx],
            'predicted_class': class_names[pred_idx],
            'count': count
        })
    
    return result

def plot_per_class_accuracy(y_true, y_pred, class_names, save_path=None):
    """
    Plot per-class accuracy
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
    
    # Sort classes by accuracy
    sorted_indices = np.argsort(class_accuracy)
    sorted_class_names = [class_names[i] for i in sorted_indices]
    sorted_accuracy = class_accuracy[sorted_indices]
    
    # Plot
    plt.figure(figsize=(15, 10))
    plt.barh(range(len(sorted_class_names)), sorted_accuracy, color='skyblue')
    plt.yticks(range(len(sorted_class_names)), sorted_class_names)
    plt.xlabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Per-class accuracy plot saved to {save_path}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate model with TTA and plot confusion matrix')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='evaluation', help='Output directory for results')
    parser.add_argument('--model_id', type=str, default='resnet152', 
                        choices=['resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 
                                'resnext101_32x8d', 'resnest50', 'resnest101', 'resnest200'], 
                        help='Model type')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], 
                        help='Dataset split to evaluate on')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create test transforms
    test_transforms = create_test_transforms(args.image_size)
    
    # We need to determine number of classes from the training data
    train_dataset = ImageDataset(root_dir=args.data_dir, split='train', transform=None)
    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    print(f"Found {num_classes} classes from training data")
    
    # Create validation dataset
    val_dataset = ImageDataset(root_dir=args.data_dir, split=args.split, transform=None)
    print(f"Loaded {len(val_dataset)} images for evaluation from {args.split} split")
    
    # Load model
    print(f"Loading {args.model_id} model")
    if "resnest" in args.model_id:
        model = torch.hub.load('zhanghang1989/ResNeSt', args.model_id, pretrained=False)
    else:
        try:
            # Try the most current way first
            model = getattr(torchvision.models, args.model_id)(pretrained=False)
        except:
            # Fallback method
            model = torch.hub.load('pytorch/vision:v0.10.0', args.model_id, pretrained=False)
    
    # Modify the final layer based on model architecture
    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        else:
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Model architecture not supported for fine-tuning")
    
    # Load weights from checkpoint
    print(f"Loading weights from checkpoint: {args.checkpoint}")
    model = load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Storage for all predictions
    all_predictions = np.zeros((len(val_dataset), num_classes))
    labels = None
    
    # Run inference with each transform
    for i, transform in enumerate(test_transforms):
        print(f"Running inference with transform {i+1}/{len(test_transforms)}")
        
        # Update dataset transform
        val_dataset.transform = transform
        
        # Create data loader
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Run inference
        predictions, batch_labels = test_with_tta(model, val_loader, device)
        
        # Store labels on first iteration
        if labels is None:
            labels = batch_labels
        
        # Accumulate predictions
        all_predictions += predictions
    
    # Average predictions
    all_predictions /= len(test_transforms)
    
    # Get predicted classes
    predicted_classes = np.argmax(all_predictions, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_classes == labels)
    print(f"\nValidation Accuracy with TTA: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    report = classification_report(labels, predicted_classes, target_names=class_names)
    print(report)
    
    # Save classification report to file
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Validation Accuracy with TTA: {accuracy:.4f}\n\n")
        f.write(report)
    
    # Find most confused classes
    top_confused = get_top_confused_classes(labels, predicted_classes, class_names)
    
    # Print and save top confused classes
    print("\nTop Confused Classes:")
    confused_file = os.path.join(args.output_dir, 'top_confused_classes.txt')
    with open(confused_file, 'w') as f:
        for item in top_confused:
            confused_info = f"True: {item['true_class']}, Predicted: {item['predicted_class']}, Count: {item['count']}"
            print(confused_info)
            f.write(confused_info + '\n')
    
    # Plot and save confusion matrix
    print("\nGenerating confusion matrix...")
    confusion_matrix_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(labels, predicted_classes, class_names, confusion_matrix_path)
    
    # Plot and save per-class accuracy
    print("Generating per-class accuracy plot...")
    accuracy_plot_path = os.path.join(args.output_dir, 'per_class_accuracy.png')
    plot_per_class_accuracy(labels, predicted_classes, class_names, accuracy_plot_path)
    
    # Save predictions and true labels for further analysis
    np.save(os.path.join(args.output_dir, 'predictions.npy'), all_predictions)
    np.save(os.path.join(args.output_dir, 'true_labels.npy'), labels)
    
    print(f"\nEvaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()