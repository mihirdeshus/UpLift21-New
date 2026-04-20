"""
generate_dataset.py
UpLift21 — Synthetic prenatal screening dataset generator.
Produces 6000 samples with realistic clinical distributions
and three-class Down syndrome risk labels.
"""

import numpy as np
import pandas as pd
from pathlib import Path

Path("data").mkdir(exist_ok=True)

np.random.seed(42)
n = 6000

# ── Clinical distributions ────────────────────────────────────────────────────
# Maternal age: normal, mean 30, sd 6, clipped 18–45
maternal_age = np.random.normal(30, 6, n).clip(18, 45)

# NT thickness: normal, mean 2.0 mm, sd 0.7
nt = np.random.normal(2.0, 0.7, n).clip(0.5, 6.0)

# CRL: normal, mean 62 mm, sd 8
crl = np.random.normal(62, 8, n).clip(45, 90)

# Beta-hCG MoM: lognormal (elevated in DS)
beta_hcg = np.random.lognormal(0.2, 0.5, n).clip(0.2, 5.0)

# PAPP-A MoM: lognormal (reduced in DS)
pappa = np.random.lognormal(-0.4, 0.4, n).clip(0.1, 3.0)

# Fetal heart rate: normal, mean 150, sd 10
fhr = np.random.normal(150, 10, n).clip(110, 180)

# Nasal bone: present in 96% of unaffected fetuses
nasal_bone = np.random.choice([1, 0], n, p=[0.96, 0.04])

data = pd.DataFrame({
    "maternal_age":  maternal_age,
    "nt_mm":         nt,
    "crl_mm":        crl,
    "beta_hcg_mom":  beta_hcg,
    "pappa_mom":     pappa,
    "fhr":           fhr,
    "nasal_bone":    nasal_bone,
})

# ── Probabilistic risk labeling ───────────────────────────────────────────────
# Weighted contribution of each marker toward DS risk score
risk_score = (
    0.35 * (data["nt_mm"] > 2.5).astype(float) +
    0.30 * (data["maternal_age"] > 35).astype(float) +
    0.20 * (data["beta_hcg_mom"] > 2.0).astype(float) +
    0.10 * (data["pappa_mom"] < 0.4).astype(float) +
    0.05 * (data["nasal_bone"] == 0).astype(float)
)

# Add biological noise
noise = np.random.normal(0, 0.05, n)
risk_prob = np.clip(risk_score + noise, 0, 1)

# Three-class labeling
#   0 = Low     (risk_prob <= 0.10)
#   1 = Moderate (0.10 < risk_prob <= 0.40)
#   2 = High    (risk_prob > 0.40)
conditions = [
    risk_prob <= 0.10,
    (risk_prob > 0.10) & (risk_prob <= 0.40),
    risk_prob > 0.40,
]
data["ds_risk_class"] = np.select(conditions, [0, 1, 2])

data.to_csv("data/prenatal_dataset.csv", index=False)

print("Dataset generated: data/prenatal_dataset.csv")
print(data["ds_risk_class"].value_counts().sort_index().rename({0: "Low", 1: "Moderate", 2: "High"}))
