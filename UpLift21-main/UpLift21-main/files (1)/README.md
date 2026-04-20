# UpLift21 v2.0 — AI-Assisted Prenatal Screening System

Frugal Science project for prenatal Down syndrome risk stratification
in low-resource clinical settings in India.

Built at Ashoka University under the Frugal Science framework
(Manu Prakash Lab, Stanford University).

---

## Project Structure

```
UpLift21/
│
├── app.py                   # Streamlit multi-page application
├── generate_dataset.py      # Synthetic dataset generator
├── train_model.py           # ML training pipeline
├── styles.css               # Dark clinical CSS theme
├── requirements.txt         # Python dependencies
├── README.md
│
├── data/
│   └── prenatal_dataset.csv
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── model_name.pkl
│   └── all_results.pkl
│
├── assets/
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   └── model_comparison.csv
│
└── reports/
    └── (PDF reports saved here)
```

---

## Setup

### 1. Create virtual environment

```bash
python -m venv venv
```

### 2. Activate

Windows (Command Prompt):
```bash
venv\Scripts\activate
```

Windows (PowerShell):
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate
```

Mac / Linux:
```bash
source venv/bin/activate
```

### 3. Create folders

```bash
mkdir data models assets reports
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run (in order)

```bash
python generate_dataset.py
python train_model.py
streamlit run app.py
```

Open: http://localhost:8501

---

## Deploy Online (Free)

1. Push folder to GitHub
2. Go to https://share.streamlit.io
3. Connect repo and deploy

---

## Model

Best performer selected automatically from:
- RandomForest
- XGBoost
- LightGBM
- CatBoost

Input features: maternal age, NT thickness, CRL, beta-hCG MoM,
PAPP-A MoM, fetal heart rate, nasal bone presence.

Output: three-class probability — Low / Moderate / High risk.

---

## Key References

- Goudarzi et al. (2025). First and Second-Trimester Screening for
  Down Syndrome: An Umbrella Review. Health Science Reports.

- Yalcin et al. (2025). AI in Prenatal Diagnosis: Down Syndrome Risk
  Assessment with Gradient Boosting. Turkish J. Obstetrics & Gynecology.

---

## Disclaimer

UpLift21 is a research prototype and decision-support tool.
It does not constitute a clinical diagnosis.
All outputs must be reviewed by a qualified clinician.
Not for use in fetal sex determination (PCPNDT Act compliance).
Requires ethics approval before any real clinical deployment.
