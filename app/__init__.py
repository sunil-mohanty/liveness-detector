from flask import Flask
#from torchvision import models
#import torch

app = Flask(__name__)
#app.config['MAX_CONTENT_LENGTH'] = 5000 * 1024 * 1024
#app.config['MAX_FORM_MEMORY_SIZE'] = 50 * 1024 * 1024  # 50 MB

# Import routes after app is initialized

# Load the ResNet model
'''
MODEL_PATH = "models/resnet_liveness.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()  # Assuming ResNet-18
model.fc = torch.nn.Linear(model.fc.in_features, 1)  # Modify for binary classification
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
'''
from app import liveness_detection
