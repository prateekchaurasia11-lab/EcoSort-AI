"""Architecture-aware EcoSort model construction and safe checkpoint loading."""
from pathlib import Path
import torch
from torch import nn
from torchvision import models,transforms
from taco_data import CLASSES


def build_model(architecture,pretrained=False):
    if architecture=='efficientnet_b0':
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model=models.efficientnet_b0(weights=weights)
        model.classifier[1]=nn.Linear(model.classifier[1].in_features,len(CLASSES))
    elif architecture=='resnet18':
        weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model=models.resnet18(weights=weights)
        model.fc=nn.Linear(model.fc.in_features,len(CLASSES))
    else: raise ValueError(f'Unsupported architecture: {architecture}')
    return model


def load_checkpoint(path,device='cpu'):
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    if checkpoint.get('class_names')!=CLASSES: raise ValueError('Unexpected class order.')
    if checkpoint.get('image_size')!=224 or checkpoint.get('mean')!=[.485,.456,.406] or checkpoint.get('std')!=[.229,.224,.225]:
        raise ValueError('Unexpected preprocessing contract.')
    architecture=checkpoint.get('architecture')
    if architecture is None:
        # Earlier verified EcoSort checkpoints have no architecture metadata.
        state=checkpoint['model_state_dict']
        if 'fc.weight' in state and 'layer4.0.conv1.weight' in state: architecture='resnet18'
        else: raise ValueError('Checkpoint has no recognizable architecture metadata.')
    model=build_model(architecture)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    model.to(device).eval()
    return model,checkpoint


def inference_transform(checkpoint):
    return transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
        transforms.Normalize(checkpoint['mean'],checkpoint['std'])])
