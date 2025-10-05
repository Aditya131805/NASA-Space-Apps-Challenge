# 🚀 NASA Space Apps Hackathon — Exoplanet & Stellar Property Prediction

## Project Overview
This project predicts exoplanet and stellar properties using machine learning. It includes both **classification** and **regression** models with evaluation metrics, visualizations, and an interactive **Streamlit UI** for exploring predictions.

The backend handles:
- Data cleaning and preprocessing
- Feature engineering
- Model training and evaluation
- Interactive visualizations

The frontend allows users to:
- Select targets for regression or classification
- Choose pre-trained or custom models
- Enter feature values for prediction
- Visualize actual vs predicted values, metrics, and feature importance

---

## Folder Structure & Files
- `models.ipynb` : Notebook containing full workflow of model training, evaluation, and visualization.
- `clean_K2.ipynb`, `clean_KOI.ipynb`, `clean_TOI.ipynb` : Data cleaning notebooks.
- `K2_cleaned.csv`, `KOI_cleaned.csv`, `TOI_cleaned.csv` : Cleaned datasets.
- `Combined Dataset.csv` : Combined dataset for training models.
- `models/` : Directory containing saved pre-trained models (`.pkl` or `.joblib` files).
- `streamlit_exoplanet_app.py` : Streamlit UI for interacting with models.
- `requirements.txt` : Python dependencies.
- `README.md` : This file.

---

## Data Sources
Datasets are from the NASA Exoplanet Archive:

- [Kepler Objects of Interest (KOI)](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)  
- [TESS Objects of Interest (TOI)](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI)  
- [K2 Planets & Candidates](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc)  

> **Note:** Only cleaned datasets are included. Raw datasets can be downloaded from the links above.

---

## Workflow

### 1. Data Cleaning
- Use `clean_K2.ipynb`, `clean_KOI.ipynb`, `clean_TOI.ipynb` to clean raw datasets.  
- Save cleaned datasets as `K2_cleaned.csv`, `KOI_cleaned.csv`, `TOI_cleaned.csv`.  

### 2. Data Combination & Preprocessing
- Combine individual datasets into `Combined Dataset.csv`.  
- Apply feature scaling and encoding for ML models.  

### 3. Model Training & Evaluation
- **Classification Target:** `disposition` (Confirmed / Candidate / False Positive)  
- **Regression Targets:** `radius`, `distance`, `equilibrium_temp`, `insolation`, `stellar_mag`, `stellar_rad`, `stellar_logg`, `stellar_teff`, `stellar_mass`  
- **Models Used:**  
  - Classification: Logistic Regression, Random Forest, KNN  
  - Regression: Linear Regression, Random Forest Regressor, KNN Regressor  
- Visualizations include confusion matrices, accuracy/metric display, correlation heatmaps, and feature importance charts.

### 4. Streamlit UI
- `streamlit_exoplanet_app.py` provides an interactive interface:
  - Load default or uploaded dataset
  - Select task: **Visualize** or **Predict**
  - Train models with selected features, test size, and random seed
  - Save & download trained models
  - Predict new values using pre-trained models
  - View metrics, feature importance, and sample predictions
  - For regression, `disposition` is automatically excluded from features

---

## Pre-trained Models
Models are saved in `models/`:

| Target | Models |
|--------|--------|
| `disposition` | Logistic Regression, Random Forest, KNN |
| `radius` | Linear Regression, Random Forest Regressor, KNN Regressor |
| `distance` | Linear Regression, Random Forest Regressor, KNN Regressor |
| `equilibrium_temp` | Linear Regression, Random Forest Regressor, KNN Regressor |
| `insolation` | Linear Regression, Random Forest Regressor, KNN Regressor |
| `stellar_mag` | Linear Regression, Random Forest Regressor, KNN Regressor |
| `stellar_rad` | Linear Regression, Random Forest Regressor, KNN Regressor |
| `stellar_logg` | Linear Regression, Random Forest Regressor, KNN Regressor |
| `stellar_teff` | Linear Regression, Random Forest Regressor, KNN Regressor |
| `stellar_mass` | Linear Regression, Random Forest Regressor, KNN Regressor |

> Regression models automatically exclude `disposition` as a feature.

---

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt

### 2. Run Streamlit App
```bash
streamlit run streamlit_exoplanet_app.py

