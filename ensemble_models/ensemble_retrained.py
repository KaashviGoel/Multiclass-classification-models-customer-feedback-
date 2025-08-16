### COMP9417 PROJECT ###
# Written 28.04.25

# Ansh Patel 	                z5478730
# Aravind Nadadur Rangarajan 	z5480898
# Tanvi Multani 	            z5621319
# Kaashvi Goel 	                z5623123
# Ash Lance 	                z5422160

# Ensemble all models together
import numpy as np
import pandas as pd
import zipfile
import os
from timeit import default_timer as timer
import warnings
import matplotlib.pyplot as plt
from matplotlib.cm import Pastel2
from collections import Counter # <-- Import Counter

# --- Preprocessing & Metrics ---
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score

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
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

# --- Predictors ---
import mlp, predict

warnings.filterwarnings('ignore')

def weighted_log_loss(y_true_ohe, y_pred_proba):
    class_counts = np.sum(y_true_ohe, axis=0)
    class_weights = 1.0 / np.maximum(class_counts, 1e-15)
    class_weights /= np.sum(class_weights)
    sample_weights = np.sum(y_true_ohe * class_weights, axis=1)

    y_pred_proba_clipped = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
    prob_true_class = np.sum(y_true_ohe * y_pred_proba_clipped, axis=1)
    
    return np.mean(-sample_weights * np.log(prob_true_class))


if __name__ == "__main__":
    # Configuration
    GROUPNAME = "z5478730"
    SELECTOR_K = 250
    OUTPUT_DIR = 'ensemble_predictions'
    VALIDATION_SIZE = 0.2
    RANDOM_STATE = 42
    NUM_CLASSES = 28 # Assuming 28 classes based on original code
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("Loading data...")
    X_train_1 = pd.read_csv("X_train.csv")
    X_train_2 = pd.read_csv("X_test_2.csv",)
    X_train_2_reduced = X_train_2.head(202)
    X_train_full = pd.concat([X_train_1, X_train_2_reduced])
    y_train_1 = pd.read_csv("y_train.csv").squeeze()
    y_train_2 = pd.read_csv("y_test_2_reduced.csv").squeeze()
    y_train_full = pd.concat([y_train_1, y_train_2])
    X_test2 = pd.read_csv("X_test_2.csv")
    X_test2.tail(1818)
    print("Data loaded.")

    # Create validation split
    print(f"Creating validation split (Size: {VALIDATION_SIZE})...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_full
    )
    print(f"Training set size: {X_tr.shape[0]}, Validation set size: {X_val.shape[0]}")

    # Fit preprocessing
    print(f"\nFitting Selector (K={SELECTOR_K}) and Scaler on training split...")
    selector = SelectKBest(f_classif, k=SELECTOR_K).fit(X_tr, y_tr)
    scaler = StandardScaler().fit(selector.transform(X_tr))
    print("Preprocessing fit completed.")

    # Validation predictions
    print("\n=== Generating Validation Predictions ===")
    val_preds_dict = {}
    models_to_run = ['lgb', 'lr', 'rf', 'xgb', 'mlp'] # Define models to include

    for model_name in models_to_run:
        print(f"\n--- Running Validation: {model_name.upper()} ---")
        if model_name == 'mlp':
             val_preds = predict.get_mlp_predictions(X_tr, y_tr, X_val, selector, scaler, val_mode=True)
        else:
             val_preds = predict.get_predictions(model_name, X_tr, y_tr, X_val, selector, scaler, val_mode=True)
        val_preds_dict[model_name] = val_preds

    # Ensemble validation predictions
    print("\nEnsembling validation predictions...")
    valid_preds_list = [pred for pred in val_preds_dict.values() if isinstance(pred, np.ndarray) and pred.shape[0] == X_val.shape[0]]

    if len(valid_preds_list) == len(models_to_run): # Check if all models produced valid predictions
        ensemble_val_preds = np.mean(valid_preds_list, axis=0)
        y_val_pred_classes = np.argmax(ensemble_val_preds, axis=1)

        # Calculate metrics
        print("\nCalculating validation metrics...")
        ohe = OneHotEncoder(sparse_output=False, categories=[np.arange(NUM_CLASSES)], handle_unknown='ignore')
        y_val_ohe = ohe.fit_transform(y_val.values.reshape(-1, 1))

        val_wll = weighted_log_loss(y_val_ohe, ensemble_val_preds)
        print(f"Weighted Log Loss (Ensemble): {val_wll:.6f}")
        val_accuracy = accuracy_score(y_val, y_val_pred_classes)
        print(f"Accuracy (Ensemble): {val_accuracy:.6f}")
        val_f1_macro = f1_score(y_val, y_val_pred_classes, average='macro')
        print(f"Macro F1-Score (Ensemble): {val_f1_macro:.6f}")

        # Plotting
        print("\nPlotting validation metrics...")
        model_names_plot = list(val_preds_dict.keys()) + ['Ensemble']
        metrics_plot = {
            'Loss': [], 'Accuracy': [], 'F1 Score': []
        }

        for name in val_preds_dict:
             pred = val_preds_dict[name]
             pred_classes = np.argmax(pred, axis=1)
             metrics_plot['Loss'].append(weighted_log_loss(y_val_ohe, pred))
             metrics_plot['Accuracy'].append(accuracy_score(y_val, pred_classes))
             metrics_plot['F1 Score'].append(f1_score(y_val, pred_classes, average='macro'))

        metrics_plot['Loss'].append(val_wll)
        metrics_plot['Accuracy'].append(val_accuracy)
        metrics_plot['F1 Score'].append(val_f1_macro)

        plt.figure(figsize=(18, 6))
        colors = Pastel2.colors
        for i, (metric, values) in enumerate(metrics_plot.items()):
            plt.subplot(1, 3, i+1)
            bars = plt.bar(model_names_plot, values, color=colors[:len(model_names_plot)])
            plt.title(f'{metric} Comparison', fontsize=14)
            plt.ylabel(metric)
            plt.xticks(rotation=45, ha='right')
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}',
                         ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig('validation_metrics.png', dpi=300, bbox_inches='tight')
        print("Validation metrics plot saved as validation_metrics.png")
        plt.show()

    else:
        print(f"Warning: Could not ensemble. Expected {len(models_to_run)} sets of predictions, got {len(valid_preds_list)}.")

    # Final training on full data
    print("\n=== Training Final Models on Full Data ===")
    print(f"Fitting Selector (K={SELECTOR_K}) and Scaler on full training data...")
    selector_final = SelectKBest(f_classif, k=SELECTOR_K).fit(X_train_full, y_train_full)
    scaler_final = StandardScaler().fit(selector_final.transform(X_train_full))
    print("Preprocessing fit completed.")

    # Store final predictions
    final_preds2 = {}

    print("\nGenerating predictions for Test Set 2...")
    for model_name in models_to_run:
         print(f"\n--- Running Final: {model_name.upper()} on Test Set 2 ---")
         # Re-use the prediction functions, they train on full data when val_mode=False
         if model_name == 'mlp':
             pred2 = predict.get_mlp_predictions(X_train_full, y_train_full, X_test2, selector_final, scaler_final, val_mode=False)
         else:
             pred2 = predict.get_predictions(model_name, X_train_full, y_train_full, X_test2, selector_final, scaler_final, val_mode=False)
         final_preds2[model_name] = pred2

    # Create ensemble predictions
    print("\nEnsembling final predictions...")
    ensemble2 = np.mean(list(final_preds2.values()), axis=0)
    print(f"Shape of final predictions for Test Set 2: {ensemble2.shape}")

    # Save results
    output_path2 = os.path.join(OUTPUT_DIR, 'preds_2.npy')
    print(f"\nSaving predictions to {output_path2}...")
    np.save(output_path2, ensemble2)
    print("Predictions saved.")

    zip_filename = f'{GROUPNAME}.zip'
    print(f"Creating submission zip file: {zip_filename}...")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
        if os.path.exists(output_path2):
             zf.write(output_path2, 'preds_2.npy')

    print("\nSubmission files created successfully!")