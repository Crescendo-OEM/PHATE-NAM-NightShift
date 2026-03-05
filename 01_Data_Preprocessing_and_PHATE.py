#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
01_Data_Preprocessing_and_PHATE.py

Description:
This script handles the initial data loading, missing value removal (complete case analysis), 
standard scaling, and constructs the nonlinear manifold using the PHATE algorithm.
It generates the global physiological topology (Figure 2).
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import phate

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Set global plot aesthetics
sns.set(style="whitegrid", font="Arial", font_scale=1.1)

# Patch for NumPy compatibility with older PHATE versions if necessary
if not hasattr(np, "float"): np.float = float
if not hasattr(np, "int"): np.int = int

# ==========================================
# 1. Define Paths and Variables
# ==========================================
DATA_DIR = "./data"
DATA_FILE = os.path.join(DATA_DIR, "dataset.xlsx")
OUTPUT_DIR = "./results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define physiological biomarkers used for latent space construction
BIO_VARS = [
"BMI", "SBP", "DBP", "AST", "ALT",
"G_GTP", "Glucose", "Total_Cholesterol", "Triglyceride", "HDL", "LDL"
]
# ==========================================
# 2. Load and Prepare Data
# ==========================================
print("Loading data...")
try:
    df = pd.read_excel(DATA_FILE)
    print(f"Original data shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Dataset not found at {DATA_FILE}. Please ensure the data file is placed correctly.")
    exit()

# Ensure the target variable exists in the dataset
if "Night_Worker" not in df.columns:
    raise ValueError("Error: 'Night_Worker' column not found in the dataset. Please ensure the target variable is included.")

# ==========================================
# 3. Preprocessing (Missing Value Removal & Scaling)
# ==========================================
print("Preprocessing data (Complete Case Analysis & Standard Scaling)...")

# Drop rows with missing values in physiological variables to ensure accurate topological mapping
cols_to_check = BIO_VARS + ["Night_Worker"]
df_clean = df.dropna(subset=cols_to_check).copy()
print(f"Data shape after removing missing values: {df_clean.shape}")

# Extract physiological variables for PHATE
X = df_clean[BIO_VARS].copy()

# Scale features to have zero mean and unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 4. Nonlinear Manifold Embedding (PHATE)
# ==========================================
print("Running PHATE algorithm to construct the latent topological space...")
# Fixed random state and stabilized hyperparameters based on sensitivity sweep
phate_op = phate.PHATE(
    knn=15,             # Adjusted for structural stability
    decay=40,           # Alpha decay parameter
    t=30,               # Optimal diffusion time
    n_landmark=None,    # Exact distance preservation (avoid landmark approximation)
    random_state=42, 
    n_jobs=-1, 
    verbose=False
)
embedding = phate_op.fit_transform(X_scaled)

# Append coordinates back to the clean dataframe
df_clean["PHATE1"] = embedding[:, 0]
df_clean["PHATE2"] = embedding[:, 1]

# Save the preprocessed dataframe for downstream analysis (LightGBM & NAM)
processed_data_path = os.path.join(DATA_DIR, "processed_dataset.csv")
df_clean.to_csv(processed_data_path, index=False)
print(f"Processed data with PHATE coordinates saved to {processed_data_path}")

# ==========================================
# 5. Visualization: Figure 2
# ==========================================
print("Generating Figure 2: Physiological distribution mapped onto the PHATE manifold...")

plt.figure(figsize=(9, 7))
sns.scatterplot(
    data=df_clean,
    x="PHATE1", 
    y="PHATE2",
    hue=df_clean["Night_Worker"].map({1: "Night Workers", 0: "Non-Night Workers"}),
    palette={"Night Workers": "#E74C3C", "Non-Night Workers": "#3498DB"},
    s=45, 
    alpha=0.8, 
    edgecolor="white", 
    linewidth=0.3
)

plt.title("Figure 2. Physiological Distribution in PHATE Latent Space", fontsize=13, fontweight="bold")
plt.xlabel("PHATE-1")
plt.ylabel("PHATE-2")
plt.legend(title="Occupational Group", fontsize=9, title_fontsize=10, loc="best")
plt.grid(alpha=0.25)
sns.despine()

# Save the figure
fig2_path = os.path.join(OUTPUT_DIR, "Figure2_PHATE_Manifold.jpg")
plt.tight_layout()
plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Figure 2 successfully saved to {fig2_path}")
print("Script 01 completed successfully.\n")


# In[ ]:




