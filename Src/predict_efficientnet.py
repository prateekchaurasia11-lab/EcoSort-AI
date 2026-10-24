"""Predict a single-item photo with architecture-specific checkpoint loading."""
import argparse
from pathlib import Path
import torch
from PIL import Image
from efficientnet_model import load_checkpoint,inference_transform


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('--model',default=str(Path(__file__).resolve().parent.parent/'Models/ecosort_efficientnet_b0.pth'))
    args=parser.parse_args()
    model,checkpoint=load_checkpoint(args.model)
    with Image.open(args.image) as image: x=inference_transform(checkpoint)(image.convert('RGB')).unsqueeze(0)
    with torch.inference_mode(): scores=model(x).softmax(1)[0]
    for i in scores.argsort(descending=True).tolist(): print(f"{checkpoint['class_names'][i]}: {scores[i]:.6%}")
    print('Scores are model outputs, not calibrated probabilities of correctness.')


if __name__=='__main__':main()
