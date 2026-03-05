#!/usr/bin/env python
# coding: utf-8

# In[8]:


"""
02_LightGBM_Risk_Mapping.py

Description:
1. Trains a LightGBM model to estimate the "Metabolic Risk Probability" based on biological markers.
2. Projects the predicted risk onto the global PHATE manifold (Figure 3).
3. Filters for the Night-shift subset to visualize occupational stress gradients (Figure 4) 
   using Kernel Density Estimation (KDE).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
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

print("Loading preprocessed data with PHATE coordinates...")
df = pd.read_csv(INPUT_FILE)

BIO_VARS = [
    "BMI", "SBP", "DBP", "AST", "ALT", 
    "G_GTP", "Glucose", "Total_Cholesterol", "Triglyceride", "HDL", "LDL"
]

# ==========================================
# 2. Risk Probability Modeling (LightGBM)
# ==========================================
print("Training LightGBM to estimate metabolic risk landscape...")

X = df[BIO_VARS]
y = df["Night_Worker"]

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "verbose": -1
}

dtrain = lgb.Dataset(X, label=y)
model = lgb.train(params, dtrain, num_boost_round=100)

df["Risk_Prob"] = model.predict(X)

# Handle cases where AUC cannot be calculated due to single-class mock data
try:
    auc_score = roc_auc_score(y, df["Risk_Prob"])
    print(f"Global Risk Mapping Completed. Model AUC: {auc_score:.3f}")
except ValueError:
    print("Global Risk Mapping Completed. (AUC skipped for small mock data)")

# ==========================================
# 3. Visualization: Figure 3 (Global Risk Map)
# ==========================================
print("Generating Figure 3: Predicted Metabolic Risk Probability Map...")

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    df["PHATE1"], df["PHATE2"], 
    c=df["Risk_Prob"], 
    cmap="coolwarm", 
    s=50, alpha=0.8, edgecolor="white", linewidth=0.3
)

cbar = plt.colorbar(scatter)
cbar.set_label("Predicted Metabolic Risk Probability", fontsize=11)
plt.title("Figure 3. Global Metabolic Risk Landscape on PHATE Manifold", fontsize=13, fontweight="bold")
plt.xlabel("PHATE-1")
plt.ylabel("PHATE-2")
plt.grid(alpha=0.2)
sns.despine()

fig3_path = os.path.join(OUTPUT_DIR, "Figure3_Global_Risk_Map.jpg")
plt.tight_layout()
plt.savefig(fig3_path, dpi=300)
plt.close()

# ==========================================
# 4. Night-Shift Subset Analysis (Figure 4)
# ==========================================
print("Transitioning to local occupational topology for night workers...")

df_night = df[df["Night_Worker"] == 1].copy()

occupational_vars = {
    "Shift_Years": "Shift Exposure Duration",
    "Consecutive_Night_Days": "Work Intensity",
    "Weekly_Work_Hours": "Weekly Labor Load"
}

available_vars = {k: v for k, v in occupational_vars.items() if k in df_night.columns}

scaler = MinMaxScaler()
for var in available_vars.keys():
    df_night[f"{var}_norm"] = scaler.fit_transform(df_night[[var]])

fig, axes = plt.subplots(1, len(available_vars), figsize=(6 * len(available_vars), 5))
if len(available_vars) == 1:
    axes = [axes]

for i, (var, label) in enumerate(available_vars.items()):
    plot_data = df_night.dropna(subset=["PHATE1", "PHATE2", f"{var}_norm"])

    try:
        if len(plot_data) > 10:
            sns.kdeplot(
                data=plot_data, x="PHATE1", y="PHATE2", weights=f"{var}_norm",
                fill=True, thresh=0.05, levels=60, cmap="coolwarm", alpha=0.85, ax=axes[i]
            )
        else:
            sns.kdeplot(
                data=plot_data, x="PHATE1", y="PHATE2",
                fill=True, cmap="coolwarm", alpha=0.85, ax=axes[i]
            )
    except Exception as e:
        axes[i].scatter(plot_data["PHATE1"], plot_data["PHATE2"], c=plot_data[f"{var}_norm"], cmap="coolwarm", s=100)

    axes[i].set_title(f"Gradient: {label}", fontsize=12, fontweight="bold")
    axes[i].set_xlabel("PHATE-1")
    axes[i].set_ylabel("PHATE-2")
    axes[i].grid(alpha=0.2)

    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[i], fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Intensity (0-1)", fontsize=10)

plt.suptitle("Figure 4. Occupational Gradient Patterns Among Night Workers", fontsize=15, fontweight="bold", y=1.05)
sns.despine()
plt.tight_layout()

fig4_path = os.path.join(OUTPUT_DIR, "Figure4_Occupational_Gradients.jpg")
plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
plt.close()

df.to_csv(INPUT_FILE, index=False)
print(f"Updated dataset with Risk Probabilities saved. Figures saved in {OUTPUT_DIR}")
print("Script 02 completed successfully.\n")


# In[ ]:




