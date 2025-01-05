import cv2
import torch
from torchvision import transforms

def preprocess_image(frame):
    """
    Preprocesses the input image for ResNet model.
    - Resizes to 224x224
    - Converts BGR to RGB
    - Normalizes as per ImageNet mean/std
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    input_tensor = transform(frame).unsqueeze(0)   # Add batch dimension
    return input_tensor
