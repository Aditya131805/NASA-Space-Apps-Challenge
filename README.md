# NASA Space Apps Hackathon - Exoplanet & Stellar Property Prediction

## Project Overview
This project predicts exoplanet and stellar properties using machine learning. It includes both **classification** and **regression** models with evaluation metrics, visualizations, and an interactive Streamlit UI for exploring predictions.

The backend handles:
- Data cleaning and preprocessing
- Feature engineering
- Model training and evaluation
- Interactive visualizations

The frontend (Streamlit) allows users to select targets, models, and visualize predictions.

---

## Folder Structure & Files
- `models.ipynb` : Notebook containing the full workflow of model training, evaluation, and visualizations.
- `clean_K2.ipynb`, `clean_KOI.ipynb`, `clean_TOI.ipynb` : Notebooks for cleaning raw datasets.
- `K2_cleaned.csv`, `KOI_cleaned.csv`, `TOI_cleaned.csv` : Cleaned individual datasets.
- `Combined Dataset.csv` : Final combined dataset used for training all models.
- `streamlit_exoplanet_app.py` : Streamlit UI for interacting with the models.
- `requirements.txt` : Python dependencies.
- `README.md` : This file.

---

## Data Sources
Raw datasets can be downloaded from the NASA Exoplanet Archive:

- [Kepler Objects of Interest (KOI)](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)  
- [TESS Objects of Interest (TOI)](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI)  
- [K2 Planets and Candidates](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc)  

> **Note:** Only cleaned datasets are included in this repository. Raw datasets can be downloaded using the links above.

---

## Workflow

### 1. Data Cleaning
- Use `clean_K2.ipynb`, `clean_KOI.ipynb`, `clean_TOI.ipynb` to clean raw datasets.  
- Save cleaned datasets as `K2_cleaned.csv`, `KOI_cleaned.csv`, `TOI_cleaned.csv`.  

### 2. Data Combination & Preprocessing
- Combine individual datasets into `Combined Dataset.csv`.  
- Apply feature scaling and encoding for machine learning models.  

### 3. Model Training & Evaluation
- **Classification Target:** `disposition` (Confirmed / Candidate / False Positive)  
- **Regression Targets:** `radius`, `distance`, `equilibrium_temp`, `insolation`, `stellar_mag`, `stellar_rad`, `stellar_logg`, `stellar_teff`, `stellar_mass`  
- **Models Used:**  
  - Classification: Logistic Regression, Random Forest, KNN  
  - Regression: Linear Regression, Random Forest Regressor, KNN Regressor  
- Visualizations include: confusion matrices, accuracy barplots, correlation heatmaps, and feature importance charts (using Seaborn).

### 4. Streamlit UI
- `streamlit_exoplanet_app.py` provides an interactive interface to:
  - Select target variable for prediction
  - Choose a model
  - Visualize actual vs predicted values
  - Compare R² / accuracy scores
  - Show feature importance charts

---

## Usage

1. Clone the repository:
```bash
git clone <your-repo-url>
