"""Prediction using the new TACO checkpoint; keeps the earlier predictor unchanged."""
import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import models, transforms


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('--model',default=str(Path(__file__).resolve().parent.parent/'Models'/'ecosort_resnet18_taco_plastic.pth'))
    args=parser.parse_args()
    checkpoint=torch.load(args.model,map_location='cpu',weights_only=True)
    classes=checkpoint['class_names']
    model=models.resnet18(weights=None)
    model.fc=torch.nn.Linear(model.fc.in_features,len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    transform=transforms.Compose([transforms.Resize((checkpoint['image_size'],checkpoint['image_size'])),
        transforms.ToTensor(),transforms.Normalize(checkpoint['mean'],checkpoint['std'])])
    with Image.open(args.image) as image:
        x=transform(image.convert('RGB')).unsqueeze(0)
    with torch.inference_mode(): scores=model(x).softmax(1)[0]
    for i in scores.argsort(descending=True).tolist(): print(f'{classes[i]}: {scores[i]:.1%}')
    print('Scores are model outputs, not calibrated probabilities of correctness.')


if __name__=='__main__': main()
