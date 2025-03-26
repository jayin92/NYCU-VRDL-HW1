import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm
from datetime import datetime

from src.datasets import ImageDataset
from src.model import create_model
from torchvision import transforms
# Import v2 transforms for CutMix and MixUp
import torchvision.transforms.v2 as transforms_v2
from torchvision.transforms import autoaugment

import wandb  # Optional for tracking experiments

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, 
                device, num_epochs=25, start_epoch=0, save_dir='checkpoints', log_wandb=False,
                use_cutmix=True, cutmix_prob=0.5, num_classes=None):
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
        start_epoch: Starting epoch (for resuming training)
        save_dir: Directory to save model checkpoints
        log_wandb: Whether to log metrics to wandb
        use_cutmix: Whether to use CutMix and MixUp augmentations
        cutmix_prob: Probability of applying CutMix/MixUp
        num_classes: Number of classes (needed for CutMix/MixUp)
    """
    # Create directory for saving checkpoints
    try:
        os.makedirs(save_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating save directory: {e}")
        raise
    
    # Initialize CutMix and MixUp transforms if enabled
    if use_cutmix and num_classes is not None:
        try:
            cutmix_transform = transforms_v2.CutMix(num_classes=num_classes)
            mixup_transform = transforms_v2.MixUp(num_classes=num_classes)
            # Randomly choose between CutMix and MixUp with equal probability
            aug_transform = transforms_v2.RandomChoice([cutmix_transform, mixup_transform])
        except Exception as e:
            print(f"Error initializing augmentation transforms: {e}")
            print("Disabling CutMix/MixUp")
            aug_transform = None
    else:
        aug_transform = None
    
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
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f'Epoch {epoch+1}/{start_epoch + num_epochs}')
        print('-' * 10)
        
        epoch_start_time = time.time()
        
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
            
            # Iterate over data
            progress_bar = tqdm(dataloader, desc=f'{phase}')
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Apply CutMix or MixUp if in training phase and enabled
                if phase == 'train' and aug_transform is not None and np.random.rand() < cutmix_prob:
                    try:
                        inputs, labels = aug_transform(inputs, labels)
                    except Exception as e:
                        print(f"Error applying augmentation: {e}")
                        # Continue without augmentation in case of error
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    # Handle both standard and CutMix/MixUp (soft) labels
                    if phase == 'train' and aug_transform is not None and len(labels.shape) > 1:
                        # For soft labels, use the class with highest probability for accuracy calculation
                        _, preds = torch.max(outputs, 1)
                        _, labels_hard = torch.max(labels, 1)
                        loss = criterion(outputs, labels)
                        batch_corrects = torch.sum(preds == labels_hard).item()
                    else:
                        # Standard processing for one-hot labels
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        batch_corrects = torch.sum(preds == labels).item()
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if scheduler is not None and isinstance(scheduler, lr_scheduler.OneCycleLR):
                            scheduler.step()
                
                # Statistics
                batch_loss = loss.item() * inputs.size(0)
                running_loss += batch_loss
                running_corrects += batch_corrects
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': batch_loss / inputs.size(0),
                    'acc': 100 * batch_corrects / inputs.size(0),
                    'lr': optimizer.param_groups[0]['lr']
                })

            # Calculate epoch statistics
            dataset_size = len(dataloader.dataset)
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects / dataset_size
            
            # Store metrics
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
                # Step the scheduler after training phase when using OneCycleLR
                
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Log to wandb if enabled
            if log_wandb:
                wandb.log({
                    f'{phase}_loss': epoch_loss,
                    f'{phase}_acc': epoch_acc,
                    'epoch': epoch,
                    'learning_rate': optimizer.param_groups[0]['lr']
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

        # Step the scheduler after each epoch when not using OneCycleLR
        if scheduler is not None and not isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step()
            print(f'LR: {scheduler.get_last_lr()}')
        
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch completed in {epoch_time:.2f}s')
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
    try:
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
    except Exception as e:
        print(f"Error plotting training history: {e}")

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
    
    try:
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
    except Exception as e:
        print(f"Error during testing: {e}")
        raise

# Custom transform that preserves aspect ratio
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

def main():
    try:
        parser = argparse.ArgumentParser(description='Train ResNet/ResNeXt/ResNeSt on Image Dataset')
        parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
        parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
        parser.add_argument('--epochs', type=int, default=80, help='Number of epochs')
        parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
        parser.add_argument('--image_size', type=int, default=224, help='Image size')
        parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint for resuming training')
        parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
        parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
        parser.add_argument('--wandb_project', type=str, default='vrdl-hw1', help='Weights & Biases project name')
        parser.add_argument('--model_id', type=str, default='resnet152', 
                            choices=['resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 
                                    'resnest50', 'resnest101', 'resnest200'], 
                            help='Model ID')
        parser.add_argument('--use_cutmix', action='store_true', help='Use CutMix and MixUp augmentations')
        parser.add_argument('--cutmix_prob', type=float, default=0.5, help='Probability of applying CutMix/MixUp')
        parser.add_argument('--preserve_aspect_ratio', action='store_true', help='Preserve aspect ratio of images')
        parser.add_argument('--scheduler', type=str, default='onecycle', choices=['cosine', 'onecycle'], 
                            help='Learning rate scheduler')
        parser.add_argument('--multi_gpu', action='store_true', help='Use multiple GPUs if available')
        args = parser.parse_args()
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Create output directories
        os.makedirs(args.output_dir, exist_ok=True)
        save_dir = os.path.join(args.output_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(save_dir, exist_ok=True)
        
        # Save args for reproducibility
        with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")
        
        # Initialize wandb if requested
        if args.use_wandb:
            try:
                wandb.init(project=args.wandb_project, config=vars(args))
            except Exception as e:
                print(f"Error initializing wandb: {e}")
                args.use_wandb = False
        
        # Define transforms
        if args.preserve_aspect_ratio:
            # Use aspect ratio preserving transform
            train_transform = transforms.Compose([
                AspectRatioPreservingTransform(args.image_size),
                transforms.RandomHorizontalFlip(),
                autoaugment.TrivialAugmentWide(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3))
            ])
        else:
            # Updated train transform
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(args.image_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                autoaugment.TrivialAugmentWide(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3))
            ])
        
        if args.preserve_aspect_ratio:
            # Validation transform with aspect ratio preservation
            val_transform = transforms.Compose([
                AspectRatioPreservingTransform(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            # Standard validation transform
            val_transform = transforms.Compose([
                transforms.Resize(int(args.image_size * 1.125)),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        # Create datasets
        try:
            train_dataset = ImageDataset(root_dir=args.data_dir, split='train', transform=train_transform)
            val_dataset = ImageDataset(root_dir=args.data_dir, split='val', transform=val_transform)
            test_dataset = ImageDataset(root_dir=args.data_dir, split='test', transform=val_transform)
        except Exception as e:
            print(f"Error creating datasets: {e}")
            raise
        
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
        
        # Load model
        print(f"Loading {args.model_id} model")
        try:
            if "resnest" in args.model_id:
                model = torch.hub.load('zhanghang1989/ResNeSt', args.model_id, pretrained=True)
            else:
                # Try the most current way first
                model = getattr(torchvision.models, args.model_id)(pretrained=True)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying alternative model loading method...")
            try:
                model = torch.hub.load('pytorch/vision', args.model_id, pretrained=True)
            except Exception as e:
                print(f"Error loading model with alternative method: {e}")
                raise
        
        # Modify the final fully connected layer for the number of classes
        num_classes = len(train_dataset.classes)
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
        
        # Use DataParallel if multi_gpu flag is set and multiple GPUs are available
        if args.multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        
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
        
        # Choose scheduler based on args
        if args.scheduler == 'onecycle':
            scheduler = lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.lr,
                steps_per_epoch=len(train_loader),
                epochs=args.epochs,
                pct_start=0.0,  # 10% of training for warmup
                div_factor=25,   # Start at max_lr/25
                final_div_factor=1000,  # End at max_lr/25000
                anneal_strategy='cos'
            )
        else:  # 'cosine'
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.lr / 1000
            )

        # Load checkpoint if specified
        start_epoch = 0
        if args.checkpoint:
            try:
                print(f"Loading checkpoint from {args.checkpoint}")
                checkpoint = torch.load(args.checkpoint, map_location=device)
                
                # Handle DataParallel models
                if isinstance(model, nn.DataParallel):
                    model.module.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if checkpoint.get('scheduler_state_dict') and scheduler is not None:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch']
                print(f"Resuming from epoch {start_epoch}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting training from scratch")
                start_epoch = 0
        
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
            num_epochs=args.epochs - start_epoch,
            start_epoch=start_epoch,
            save_dir=save_dir,
            log_wandb=args.use_wandb,
            use_cutmix=args.use_cutmix,
            cutmix_prob=args.cutmix_prob,
            num_classes=num_classes
        )
        
        # Test the model
        print("Testing the model...")
        test_acc = test_model(model, test_loader, device)
        
        # Save final model
        final_model_path = os.path.join(save_dir, 'final_model.pt')
        if isinstance(model, nn.DataParallel):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
            
        torch.save({
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'test_acc': test_acc,
            'history': history,
            'classes': train_dataset.classes
        }, final_model_path)
        print(f"Saved final model to {final_model_path}")
        
        if args.use_wandb:
            try:
                wandb.log({'test_accuracy': test_acc})
                wandb.finish()
            except Exception as e:
                print(f"Error logging to wandb: {e}")
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        if args.use_wandb:
            try:
                wandb.finish()
            except:
                pass

if __name__ == "__main__":
    main()