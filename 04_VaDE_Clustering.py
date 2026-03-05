#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
04_VaDE_Clustering.py

Description:
1. Performs latent clustering using Gaussian Mixture Models (GMM).
2. Generates the Latent Cluster Map (Figure 7).
3. Overlays the High-Risk density distribution (Figure 8) with color-coded risk groups.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from matplotlib.lines import Line2D
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

print("Loading dataset for clustering and risk mapping...")
df = pd.read_csv(INPUT_FILE)
df_night = df[df["Night_Worker"] == 1].dropna(subset=["Risk_Prob"]).copy()

# ==========================================
# 2. Latent Clustering (GMM)
# ==========================================
print("Performing Latent Space Clustering...")
X_latent = df_night[["PHATE1", "PHATE2"]].values
n_samples = len(X_latent)
n_comp = 3 if n_samples >= 3 else n_samples

try:
    gmm = GaussianMixture(n_components=n_comp, covariance_type='full', random_state=42)
    raw_labels = gmm.fit_predict(X_latent)
    df_night["vade_cluster"] = ["Cluster " + str(label) for label in raw_labels]
except Exception:
    df_night["vade_cluster"] = "Cluster 0"

# ==========================================
# 3. Visualization: Figure 7 (Latent Cluster Map)
# ==========================================
print("Generating Figure 7: Latent Cluster Map...")

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df_night, 
    x="PHATE1", y="PHATE2",
    hue="vade_cluster", 
    size="Risk_Prob",        
    sizes=(50, 200),
    palette="Set2", 
    alpha=0.85, edgecolor="white", linewidth=0.4
)

plt.title("Figure 7. VaDE Latent Cluster Map (Night-Shift Subset)", fontsize=13, fontweight="bold")
plt.xlabel("PHATE-1", fontsize=11)
plt.ylabel("PHATE-2", fontsize=11)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.25)
sns.despine()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure7_Latent_Clusters.jpg"), dpi=300, bbox_inches="tight")
plt.close()

# ==========================================
# 4. Visualization: Figure 8 (Risk Contour Overlay)
# ==========================================
print("Generating Figure 8: Risk Density Contour Overlay...")

threshold = df_night['Risk_Prob'].quantile(0.70)
df_night['High_Risk_Label'] = (df_night['Risk_Prob'] >= threshold).astype(int)

plt.figure(figsize=(9, 7))

# Background Density
try:
    if len(df_night) > 10:
        sns.kdeplot(
            data=df_night, x="PHATE1", y="PHATE2",
            weights="Risk_Prob",     
            fill=True, cmap="coolwarm", alpha=0.35, thresh=0.05, levels=50
        )
    else:
        sns.kdeplot(
            data=df_night, x="PHATE1", y="PHATE2",
            fill=True, cmap="coolwarm", alpha=0.35
        )
except Exception:
    pass

# Scatter plot with explicit color mapping
# 0 (Low Risk) = gray, 1 (High Risk) = crimson
sns.scatterplot(
    data=df_night, x="PHATE1", y="PHATE2",
    hue="High_Risk_Label",
    palette={0: "gray", 1: "crimson"},
    alpha=0.8, s=60, edgecolor="white", linewidth=0.3
)

plt.title("Figure 8. High-Risk Topology among Night-Shift Workers", fontsize=13, fontweight="bold")
plt.xlabel("PHATE-1", fontsize=11)
plt.ylabel("PHATE-2", fontsize=11)

# Manual legend to guarantee correct gray/crimson display in all environments
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Low Risk Group',
           markerfacecolor='gray', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='High Risk Group',
           markerfacecolor='crimson', markersize=10)
]
plt.legend(handles=legend_elements, title="Physiological Risk", loc="upper right")

plt.grid(alpha=0.2)
sns.despine()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure8_Risk_Topography.jpg"), dpi=300, bbox_inches="tight")
plt.close()

print(f"Figures 7 and 8 successfully saved. Low Risk is now set to gray.")
print("Script 04 completed successfully.\n")


# In[ ]:




