from pathlib import Path
import argparse

import torch
from PIL import Image
from torchvision import models, transforms


def main():
    parser = argparse.ArgumentParser(description="Predict a waste image.")
    parser.add_argument("image", help="Path to the image")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        parser.error(f"Image not found: {image_path}")

    project_dir = Path(__file__).resolve().parent.parent
    model_path = project_dir / "Models" / "ecosort_resnet18_wrappers.pth"

    checkpoint = torch.load(
        model_path, map_location="cpu", weights_only=True
    )
    classes = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    with Image.open(image_path) as image:
        image_tensor = transform(image.convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        scores = model(image_tensor).softmax(dim=1)[0]

    best = scores.argmax().item()
    print(f"\nPrediction: {classes[best]}")
    print(f"Model score: {scores[best].item():.1%}")
    print("\nAll class scores:")
    for index in scores.argsort(descending=True).tolist():
        print(f"  {classes[index]}: {scores[index].item():.1%}")


if __name__ == "__main__":
    main()