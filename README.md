# EcoSort-AI


## E2 MobileNetV2 Inference

E2 is maintained separately from the existing PyTorch ResNet18 predictor.

Install dependencies:
    pip install -r requirements-e2-inference.txt

Run prediction:
    python Src/predict_keras.py image.jpg --checkpoint path/to/best.keras --protocol path/to/protocol.json

The checkpoint and protocol must be stored outside Git.

Input: RGB, 224x224, pixel values 0-255.
Rescaling: embedded in the model.
Model scores are uncalibrated.
