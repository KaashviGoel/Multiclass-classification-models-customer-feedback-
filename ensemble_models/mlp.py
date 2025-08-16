### COMP9417 PROJECT ###
# Written 28.04.25

# Ansh Patel 	                z5478730
# Aravind Nadadur Rangarajan 	z5480898
# Tanvi Multani 	            z5621319
# Kaashvi Goel 	                z5623123
# Ash Lance 	                z5422160

# mlp model for ensemble code

# --- PyTorch MLP ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3, num_classes=28):
        super().__init__()
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(current_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU(), nn.Dropout(dropout)])
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, num_classes))
        self.layers = nn.Sequential(*layers)
    def forward(self, x): return self.layers(x)

def focal_loss(logits, targets, gamma=2.0):
    ce_loss = F.cross_entropy(logits, targets, reduction='none'); pt = torch.exp(-ce_loss)
    return ((1 - pt) ** gamma * ce_loss).mean()