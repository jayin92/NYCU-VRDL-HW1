import torch
import torch.nn as nn
import torchvision.models as models
import math

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class FPNBlock(nn.Module):
    def __init__(self, lateral_channel, out_channel):
        super(FPNBlock, self).__init__()
        self.lateral_conv = nn.Conv2d(lateral_channel, out_channel, kernel_size=1, stride=1, padding=0)
        self.fpn_conv = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x, up_x=None):
        lateral = self.lateral_conv(x)
        
        if up_x is not None:
            # Upsample and add
            up_size = (lateral.shape[2], lateral.shape[3])
            up = nn.functional.interpolate(up_x, size=up_size, mode='nearest')
            lateral = lateral + up
            
        return self.fpn_conv(lateral)

class SEResNet50FPN(nn.Module):
    def __init__(self, num_classes, pretrained=True, fpn_channels=256):
        """
        SE-ResNet50 model with Feature Pyramid Network
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
            fpn_channels (int): Number of channels in FPN
        """
        super(SEResNet50FPN, self).__init__()
        
        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Extract ResNet50 layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # Layer 1 (C2)
        self.layer1 = resnet.layer1
        self.se1 = SELayer(256)
        
        # Layer 2 (C3)
        self.layer2 = resnet.layer2
        self.se2 = SELayer(512)
        
        # Layer 3 (C4)
        self.layer3 = resnet.layer3
        self.se3 = SELayer(1024)
        
        # Layer 4 (C5)
        self.layer4 = resnet.layer4
        self.se4 = SELayer(2048)
        
        # FPN blocks for each level
        self.fpn_c5 = FPNBlock(2048, fpn_channels)
        self.fpn_c4 = FPNBlock(1024, fpn_channels)
        self.fpn_c3 = FPNBlock(512, fpn_channels)
        self.fpn_c2 = FPNBlock(256, fpn_channels)
        
        # Extra FPN level (P6)
        self.fpn_p6 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, stride=2, padding=1)
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Add SE layer to each FPN output
        self.se_p5 = SELayer(fpn_channels)
        self.se_p4 = SELayer(fpn_channels)
        self.se_p3 = SELayer(fpn_channels)
        self.se_p2 = SELayer(fpn_channels)
        self.se_p6 = SELayer(fpn_channels)
        
        # Classifier with concatenated features from all FPN levels
        self.fc = nn.Linear(fpn_channels * 5, num_classes)
        
    def forward(self, x):
        # ResNet50 backbone with SE blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c2 = self.layer1(x)
        c2 = self.se1(c2)
        
        c3 = self.layer2(c2)
        c3 = self.se2(c3)
        
        c4 = self.layer3(c3)
        c4 = self.se3(c4)
        
        c5 = self.layer4(c4)
        c5 = self.se4(c5)
        
        # FPN feature maps
        p5 = self.fpn_c5(c5)
        p4 = self.fpn_c4(c4, p5)
        p3 = self.fpn_c3(c3, p4)
        p2 = self.fpn_c2(c2, p3)
        
        # Extra level P6
        p6 = self.fpn_p6(p5)
        
        # Apply SE blocks to FPN outputs
        p5 = self.se_p5(p5)
        p4 = self.se_p4(p4)
        p3 = self.se_p3(p3)
        p2 = self.se_p2(p2)
        p6 = self.se_p6(p6)
        
        # Global pooling on each FPN level
        p2_feat = self.avgpool(p2).flatten(1)
        p3_feat = self.avgpool(p3).flatten(1)
        p4_feat = self.avgpool(p4).flatten(1)
        p5_feat = self.avgpool(p5).flatten(1)
        p6_feat = self.avgpool(p6).flatten(1)
        
        # Concatenate features from all levels
        feat = torch.cat([p2_feat, p3_feat, p4_feat, p5_feat, p6_feat], dim=1)
        
        # Classification
        x = self.fc(feat)
        
        return x

class ResNet512(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        """
        ResNet-152 model with 512x512 input size
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
        """
        super(ResNet512, self).__init__()
        
        # Load pretrained ResNet-152
        self.resnet = models.resnet152(pretrained=pretrained)
        
        # Replace the final fully connected layer to match number of classes
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

def create_model(num_classes, pretrained=True, model_type="resnet152"):
    """
    Create and initialize the model
    
    Args:
        num_classes (int): Number of classes for classification
        pretrained (bool): Whether to use pretrained weights
        model_type (str): Type of model to create ("resnet152", "seresnet50fpn", etc.)
    
    Returns:
        model: Initialized model
    """
    if model_type == "seresnet50fpn":
        model = SEResNet50FPN(num_classes=num_classes, pretrained=pretrained)
    else:  # default to resnet152
        model = ResNet512(num_classes=num_classes, pretrained=pretrained)
    return model

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
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# Example usage
if __name__ == "__main__":
    # Create models with 10 classes (example)
    model_resnet152 = create_model(num_classes=10, model_type="resnet152")
    model_seresnet50fpn = create_model(num_classes=10, model_type="seresnet50fpn")
    
    # Print model architectures
    print("ResNet-152 model:")
    print(model_resnet152)
    print("\nSE-ResNet-50 FPN model:")
    print(model_seresnet50fpn)
    
    # Test with a random input tensor (batch_size=1, channels=3, height=512, width=512)
    dummy_input = torch.randn(1, 3, 512, 512)
    
    output_resnet152 = model_resnet152(dummy_input)
    output_seresnet50fpn = model_seresnet50fpn(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"ResNet-152 output shape: {output_resnet152.shape}")
    print(f"SE-ResNet-50 FPN output shape: {output_seresnet50fpn.shape}")
    
    # Calculate number of parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nResNet-152 parameter count: {count_parameters(model_resnet152):,}")
    print(f"SE-ResNet-50 FPN parameter count: {count_parameters(model_seresnet50fpn):,}")