### COMP9417 PROJECT ###
# Written 28.04.25

# Ansh Patel 	                z5478730
# Aravind Nadadur Rangarajan 	z5480898
# Tanvi Multani 	            z5621319
# Kaashvi Goel 	                z5623123
# Ash Lance 	                z5422160

# This code writes a MLP with Focal Loss 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[2], 28)
        )

    def forward(self, x):
        return self.layers(x)

    def focal_loss(logits, targets, gamma=2.0):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** gamma * ce_loss).mean()

def model(X_train, y_train):
    X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

    selector_k = 250
    selector = SelectKBest(score_func=f_classif, k=selector_k)
    X_tr = selector.fit_transform(X_tr, y_tr)
    X_val = selector.transform(X_val)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(input_dim=selector_k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)

    # Fit scalar 
    scaler = StandardScaler()
    scaler.fit(X_tr)
    X_tr_scaled = scaler.transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    X_tensor = torch.tensor(X_tr_scaled, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_tr.values, dtype=torch.long).to(device)
    train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=128, shuffle=True)

    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)

    best_f1 = 0
    patience = 5
    wait = 0

    for epoch in range(30):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = MLP.focal_loss(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_f1 = f1_score(y_val, val_preds, average='macro')
            print(f"Epoch {epoch+1}, Val F1: {val_f1:.4f}")
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model = model.state_dict()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping.")
                    break

    model.load_state_dict(best_model)

    mlp_val_preds = torch.argmax(model(X_val_tensor), dim=1).cpu().numpy()
    report = classification_report(y_val, mlp_val_preds, zero_division=0)
    return model, y_val, mlp_val_preds, best_f1, report

