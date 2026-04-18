import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

def load_model():
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)
    model.eval()
    return model