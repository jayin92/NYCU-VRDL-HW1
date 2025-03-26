import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

from src.datasets import ImageDataset
from src.model import create_model
from torchvision import transforms
import wandb  # Optional for tracking experiments

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, 
                device, num_epochs=25, save_dir='checkpoints', log_wandb=False):
    """
    Train the model and validate
    
    Args:
        model: PyTorch model
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on (cuda/cpu)
        num_epochs: Number of epochs to train
        save_dir: Directory to save model checkpoints
        log_wandb: Whether to log metrics to wandb
    """
    # Create directory for saving checkpoints
    os.makedirs(save_dir, exist_ok=True)
    
    best_acc = 0.0
    best_model_path = None
    
    # Keep track of metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Store start time for calculation of training duration
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluation mode
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            progress_bar = tqdm(dataloader, desc=f'{phase}')
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                
                # Statistics
                batch_loss = loss.item() * inputs.size(0)
                batch_corrects = torch.sum(preds == labels.data).item()
                running_loss += batch_loss
                running_corrects += batch_corrects
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': batch_loss / inputs.size(0),
                    'acc': 100 * batch_corrects / inputs.size(0)
                })

            # if phase == 'train' and scheduler is not None:
                # scheduler.step()
            
            # Calculate epoch statistics
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects / len(dataloader.dataset)
            
            # Store metrics
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Log to wandb if enabled
            if log_wandb:
                wandb.log({
                    f'{phase}_loss': epoch_loss,
                    f'{phase}_acc': epoch_acc,
                    'epoch': epoch
                })
            
            # Save the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch+1}_acc_{epoch_acc:.4f}.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'loss': epoch_loss,
                    'acc': epoch_acc,
                }, checkpoint_path)
                
                best_model_path = checkpoint_path
                print(f'Saved new best model to {checkpoint_path}')
        
        # Save checkpoint every 5 epochs regardless of performance
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': epoch_loss,
                'acc': epoch_acc,
            }, checkpoint_path)
            print(f'Saved checkpoint to {checkpoint_path}')
        
        print()
    
    # Calculate total training time
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f} (model saved at: {best_model_path})')
    
    # Plot training history
    plot_training_history(history, save_dir)
    
    return model, history

def plot_training_history(history, save_dir):
    """
    Plot training and validation loss and accuracy
    
    Args:
        history: Dictionary containing training metrics
        save_dir: Directory to save the plots
    """
    # Plot loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

def test_model(model, test_loader, device):
    """
    Test the model on test dataset
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device to test on (cuda/cpu)
    
    Returns:
        accuracy: Test accuracy
    """
    model.eval()
    correct = 0
    total = 0
    
    class_correct = [0] * len(test_loader.dataset.classes)
    class_total = [0] * len(test_loader.dataset.classes)
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Calculate per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    # Overall accuracy
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    
    # Per-class accuracy
    for i in range(len(test_loader.dataset.classes)):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i]
            print(f'Accuracy of {test_loader.dataset.classes[i]}: {class_acc:.4f}')
    
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Train SE-ResNet50 on Image Dataset')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--image_size', type=int, default=640, help='Image size')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint for resuming training')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='vrdl-hw1', help='Weights & Biases project name')
    parser.add_argument('--model_id', type=str, default='resnet152', choices=['resnet152', 'resnext101_32x8d']) # SE 
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    save_dir = os.path.join(args.output_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(0.6, 1.0)),  # Larger scale to preserve details
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),  # Plants can be viewed from different angles
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1),  # More saturation variation for plants
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),  # For robustness to blur
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3))  # Random erasing for regularization
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(int(args.image_size * 1.125)),  # Resize the smallest side to 576 (if image_size=512)
        transforms.CenterCrop(args.image_size),           # Then center crop
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ImageDataset(root_dir=args.data_dir, split='train', transform=train_transform)
    val_dataset = ImageDataset(root_dir=args.data_dir, split='val', transform=val_transform)
    test_dataset = ImageDataset(root_dir=args.data_dir, split='test', transform=val_transform)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Load model from torch.hub
    print(f"Loading {args.model_id} model")
    # model = create_model(num_classes=len(train_dataset.classes), model_type=args.model_type)
    # model = torch.hub.load('moskomule/senet.pytorch', 'se_resnet101',     =True)
    model = torch.hub.load('pytorch/vision:v0.10.0', args.model_id, pretrained=True)
    
    # Modify the final fully connected layer for the number of classes
    num_classes = len(train_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Move model to device
    model = model.to(device)
    
    # Print model summary
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Define loss function, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler (cosine annealing with warm restarts)
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, 
    #     T_0=10,  # Restart every 10 epochs
    #     T_mult=1, 
    #     eta_min=args.lr / 100
    # )

    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,  # Maximum learning rate
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.3,  # Spend 30% of training time in the increasing phase
        div_factor=25,  # Initial learning rate will be max_lr/25
        final_div_factor=10000,  # Final learning rate will be max_lr/10000
        anneal_strategy='cos'  # Use cosine annealing
    )
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}")
    
    # Train the model
    print("Starting training...")
    model, history = train_model(
        model, 
        criterion, 
        optimizer, 
        scheduler, 
        train_loader, 
        val_loader, 
        device, 
        num_epochs=args.epochs, 
        save_dir=save_dir,
        log_wandb=args.use_wandb
    )
    
    # Test the model
    print("Testing the model...")
    test_acc = test_model(model, test_loader, device)
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'test_acc': test_acc,
        'history': history,
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    if args.use_wandb:
        wandb.log({'test_accuracy': test_acc})
        wandb.finish()

if __name__ == "__main__":
    main()