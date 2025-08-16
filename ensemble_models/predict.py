### COMP9417 PROJECT ###
# Written 28.04.25

# Ansh Patel 	                z5478730
# Aravind Nadadur Rangarajan 	z5480898
# Tanvi Multani 	            z5621319
# Kaashvi Goel 	                z5623123
# Ash Lance 	                z5422160

# classic models for ensemble

from collections import Counter
import numpy as np
import pandas as pd
import zipfile
import os
from timeit import default_timer as timer
import warnings
import traceback # Import traceback at the top

# --- Preprocessing & Metrics ---
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.model_selection import train_test_split # Now used for local validation split
from sklearn.metrics import classification_report, f1_score, accuracy_score # Added accuracy

# --- Model Classes ---
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- PyTorch MLP ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# --- Imbalance Handling ---
try:
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    print("Warning: imbalanced-learn not found. SMOTE will not be used for LR/RF.")
    SMOTE_AVAILABLE = False

import mlp

# Wrapper functions for predictions (ensemble)

model_configs = {
    'lgb': {
        'model': LGBMClassifier(class_weight='balanced', num_class=28, boosting_type='gbdt',
                               learning_rate=0.05, n_estimators=200, max_depth=8,
                               num_leaves=64, subsample=0.8, colsample_bytree=0.8,
                               random_state=42, verbosity=-1),
        'resample': False
    },
    'lr': {
        'model': LogisticRegression(max_iter=2000, class_weight='balanced',
                                  C=0.01, penalty='l2', random_state=42, n_jobs=-1),
        'resample': True
    },
    'rf': {
        'model': RandomForestClassifier(n_estimators=250, max_depth=15, min_samples_split=8,
                                      min_samples_leaf=4, class_weight='balanced_subsample',
                                      random_state=42, n_jobs=-1),
        'resample': True
    },
    'xgb': {
        'model': XGBClassifier(objective='multi:softprob', num_class=28, eval_metric='mlogloss',
                              n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8,
                              colsample_bytree=0.8, min_child_weight=3, gamma=0.2,
                              random_state=42, use_label_encoder=False),
        'resample': False
    }
}

# --- get_predictions function ---
def get_predictions(model_name, X_train, y_train, X_pred, selector, scaler, val_mode=False):
    start_time = timer()
    cfg = model_configs[model_name]


    # Feature processing
    X_train_processed = scaler.transform(selector.transform(X_train))
    X_pred_processed = scaler.transform(selector.transform(X_pred))

    # Handle resampling
    if cfg['resample'] and val_mode:
        # Calculate minimum class size in the current training fold (y_train corresponds to y_tr here)
        class_counts = Counter(y_train)
        min_class_count = min(class_counts.values()) if class_counts else 0

        # SMOTE's k_neighbors must be >= 1 and < number of samples in the smallest class.
        # Default k_neighbors is 5. Need at least 6 samples for default.
        if min_class_count > 1: # Check if resampling is possible
            # Adjust k_neighbors: must be at least 1 and less than min_class_count
            # Cap k_neighbors at the original desired value (5) if possible
            smote_k = min(5, min_class_count - 1)
            smote_k = max(1, smote_k) # Ensure k is at least 1

            print(f"Resampling for {model_name.upper()} with SMOTE k_neighbors={smote_k} (min class size: {min_class_count})")
            resampler = SMOTETomek(smote=SMOTE(k_neighbors=smote_k, random_state=42),
                                   random_state=42, n_jobs=-1)
            X_res, y_res = resampler.fit_resample(X_train_processed, y_train)
        else:
            # If min_class_count is 0 or 1, SMOTE cannot run. Skip resampling.
            print(f"Skipping resampling for {model_name.upper()}: minimum class size ({min_class_count}) is too small.")
            X_res, y_res = X_train_processed, y_train
    else: # No resampling if cfg['resample'] is False or if not in val_mode
        X_res, y_res = X_train_processed, y_train
        # Optional print for clarity during final training
        # if cfg['resample'] and not val_mode:
        #     print(f"Skipping resampling for {model_name.upper()} during final training.")

    # Train model
    model = cfg['model']
    model.fit(X_res, y_res)

    # Generate predictions
    probs = model.predict_proba(X_pred_processed)
    print(f"{model_name.upper()} completed in {timer()-start_time:.1f}s")
    return probs

def get_mlp_predictions(X_train, y_train, X_pred, selector, scaler, val_mode=False):
    start_time = timer()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Feature processing
    X_train_processed = scaler.transform(selector.transform(X_train))
    X_pred_processed = scaler.transform(selector.transform(X_pred))

    # Convert to tensors
    X_tensor = torch.tensor(X_train_processed, dtype=torch.float32).to(device)
    # Ensure y_train is suitable for tensor conversion (e.g., Series or numpy array)
    y_tensor = torch.tensor(y_train.values if isinstance(y_train, pd.Series) else y_train, dtype=torch.long).to(device)
    train_dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Determine num_classes dynamically
    num_classes = len(np.unique(y_train))

    # Initialize and train model
    model = mlp.MLP(X_train_processed.shape[1], num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    epochs = 25 # Number of epochs
    print(f"Training MLP for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        # Optional: add epoch loss tracking if needed
        # epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            # epoch_loss += loss.item()
        # Optional: print epoch loss
        # print(f"MLP Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")


    # Generate predictions
    model.eval()
    with torch.no_grad():
        X_pred_tensor = torch.tensor(X_pred_processed, dtype=torch.float32).to(device)
        logits = model(X_pred_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()

    print(f"MLP completed in {timer()-start_time:.1f}s")
    return probs