#!/usr/bin/env python
# coding: utf-8

# In[4]:


"""
03_NAM_Deep_Learning.py

Description:
1. Filters the dataset to include ONLY night-shift workers (Target Subset).
2. Defines high-risk workers as the top 30% based on the LightGBM Risk Probability.
3. Trains an interpretable Neural Additive Model (NAM) using PyTorch.
4. Generates model performance metrics (Figure 6) and feature interpretations (Figure 5).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from sklearn.calibration import calibration_curve
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")
sns.set(style="whitegrid", font="Arial", font_scale=1.1)

# ==========================================
# 1. Define Paths and Load Processed Data
# ==========================================
DATA_DIR = "./data"
INPUT_FILE = os.path.join(DATA_DIR, "processed_dataset.csv")
OUTPUT_DIR = "./results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset with Risk Probabilities...")
df = pd.read_csv(INPUT_FILE)

# ==========================================
# 2. Night-Shift Subset Filtering & Target Definition
# ==========================================
print("Filtering for Night-Shift Subset...")
df_night = df[df["Night_Worker"] == 1].copy()
df_night = df_night.dropna(subset=['Risk_Prob'])

threshold = df_night['Risk_Prob'].quantile(0.70)
df_night['High_Risk'] = (df_night['Risk_Prob'] >= threshold).astype(int)

print(f"Subset shape: {df_night.shape}")

FEATURE_COLS = [
    'BMI', 'SBP', 'DBP', 'AST', 'ALT', 'G_GTP', 'Glucose',
    'Total_Cholesterol', 'Triglyceride', 'HDL', 'LDL',
    'Shift_Years', 'Consecutive_Night_Days', 'Weekly_Work_Hours',
    'PHATE1', 'PHATE2'
]

available_features = [col for col in FEATURE_COLS if col in df_night.columns]
df_model = df_night.dropna(subset=available_features).copy()

X = df_model[available_features]
y = df_model['High_Risk']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.5, random_state=42
    )

# ==========================================
# 3. Neural Additive Model (NAM) Architecture
# ==========================================
class NAMLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x.unsqueeze(-1))

class NAMModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.ModuleList([NAMLayer() for _ in range(input_dim)])
        self.out = nn.Linear(input_dim, 1)

    def forward(self, x):
        outs = [layer(x[:, i]) for i, layer in enumerate(self.layers)]
        concat = torch.cat(outs, dim=1)
        return torch.sigmoid(self.out(concat))

# ==========================================
# 4. Model Training
# ==========================================
print("Training Neural Additive Model (NAM)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NAMModel(input_dim=X_train.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_t)
    loss = criterion(y_pred, y_train_t)
    loss.backward()
    optimizer.step()

# ==========================================
# 5. Model Evaluation (Figure 6)
# ==========================================
print("Evaluating Model Performance...")
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_t).cpu().numpy().flatten()

try:
    auc_val = roc_auc_score(y_test, y_pred_test)
    auc_label = f"AUC = {auc_val:.3f}"
    print(f"NAM Performance -> AUC: {auc_val:.3f}")
except ValueError:
    auc_label = "AUC = N/A (Insufficient Data)"
    print("Notice: Performance metrics skipped due to small mock data size.")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Figure 6a. ROC Curve
try:
    fpr, tpr, _ = roc_curve(y_test, y_pred_test)
    axes[0].plot(fpr, tpr, color='darkred', lw=2, label=auc_label)
except ValueError:
    axes[0].text(0.5, 0.5, 'Insufficient data to plot ROC', ha='center', va='center')

axes[0].plot([0, 1], [0, 1], '--', color='gray')
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("Figure 6a. ROC Curve (NAM)", fontweight='bold')
axes[0].legend(loc="lower right", frameon=False)

# Figure 6b. Calibration Curve
try:
    prob_true, prob_pred = calibration_curve(y_test, y_pred_test, n_bins=10)
    axes[1].plot(prob_pred, prob_true, "o-", color='navy', label='NAM Calibration')
except ValueError:
    axes[1].text(0.5, 0.5, 'Insufficient data for calibration', ha='center', va='center')

axes[1].plot([0, 1], [0, 1], '--', color='gray')
axes[1].set_xlabel("Predicted Probability")
axes[1].set_ylabel("Observed Frequency")
axes[1].set_title("Figure 6b. Calibration Curve", fontweight='bold')
axes[1].legend(loc="upper left", frameon=False)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure6_NAM_Performance.jpg"), dpi=300)
plt.close()

# ==========================================
# 6. Feature Contributions (Figure 5a)
# ==========================================
print("Extracting Feature Contributions...")
with torch.no_grad():
    contribs = []
    for i, col in enumerate(available_features):
        single_input = torch.zeros_like(X_test_t)
        single_input[:, i] = X_test_t[:, i]
        out = model.layers[i](single_input[:, i])
        contribs.append(out.mean().item())

feature_importance = pd.DataFrame({
    'Feature': available_features,
    'Contribution': contribs
}).sort_values('Contribution', ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(data=feature_importance, x='Contribution', y='Feature', palette='coolwarm')
plt.title("Figure 5a. NAM Feature Contributions", fontweight='bold')
plt.xlabel("Mean Output Contribution")
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure5a_NAM_Contributions.jpg"), dpi=300)
plt.close()

# ==========================================
# 7. Partial Dependence Plots (Figure 5b, c, d)
# ==========================================
print("Generating Partial Dependence Plots...")
pdp_features = ['Triglyceride', 'Weekly_Work_Hours', 'PHATE2']
pdp_labels = ['5b', '5c', '5d']

for feat, label in zip(pdp_features, pdp_labels):
    if feat in available_features:
        feat_idx = available_features.index(feat)
        X_grid = X_test.copy()
        feat_values = np.linspace(X_test[:, feat_idx].min(), X_test[:, feat_idx].max(), 100)
        avg_preds = []
        for val in feat_values:
            X_grid[:, feat_idx] = val
            with torch.no_grad():
                preds = model(torch.tensor(X_grid, dtype=torch.float32).to(device)).cpu().numpy().flatten()
            avg_preds.append(np.mean(preds))
        plt.figure(figsize=(5, 4))
        plt.plot(feat_values, avg_preds, color='crimson', linewidth=2)
        plt.title(f"Figure {label}. PDP: {feat}", fontweight='bold')
        plt.xlabel(f"{feat} (Standardized)")
        plt.ylabel("Predicted Risk Probability")
        plt.grid(alpha=0.3)
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"Figure{label}_NAM_PDP_{feat}.jpg"), dpi=300)
        plt.close()

print("Script 03 completed successfully.")


# In[ ]:




