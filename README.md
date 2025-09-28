# NASA Space Apps Hackathon - Exoplanet Prediction

## Project Overview
This project predicts exoplanet and stellar properties using machine learning. It includes classification and regression models with evaluation metrics and visualizations. The backend handles data cleaning, feature engineering, model training, and evaluation.

## Folder Structure & Files
- `main.ipynb` : Notebook containing full workflow of model training, evaluation, and visualizations.
- `clean_K2.ipynb`, `clean_KOI.ipynb`, `clean_TOI.ipynb` : Notebooks for cleaning raw datasets.
- `K2_cleaned.csv`, `KOI_cleaned.csv`, `TOI_cleaned.csv` : Cleaned individual datasets.
- `Combined Dataset.csv` : Final combined dataset used for training all models.
- `models.ipynb` : Notebook with detailed comparison of models for each target.
- `requirements.txt` : Python dependencies.
- `README.md` : This file.

## Data Sources
Raw datasets can be downloaded from the NASA Exoplanet Archive:

- [Kepler Objects of Interest (KOI)](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)  
- [TESS Objects of Interest (TOI)](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI)  
- [K2 Planets and Candidates](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc)  

> **Note:** Only cleaned datasets are included in this repository. Raw datasets can be downloaded using the links above.

## Workflow
1. **Data Cleaning**  
   - Use `clean_K2.ipynb`, `clean_KOI.ipynb`, `clean_TOI.ipynb` to clean raw datasets.  
   - Save cleaned datasets as `K2_cleaned.csv`, `KOI_cleaned.csv`, `TOI_cleaned.csv`.  

2. **Data Combination & Preprocessing**  
   - Combine individual datasets into `Combined Dataset.csv`.  
   - Apply feature scaling and encoding for machine learning models.  

3. **Model Training & Evaluation**  
   - Classification targets: `disposition`  
   - Regression targets: `radius`, `distance`, `equilibrium_temp`, `insolation`, `stellar_mag`, `stellar_rad`, `stellar_logg`, `stellar_teff`, `stellar_mass`  
   - Models used: Logistic Regression, Random Forest, KNN, Linear Regression, Random Forest Regressor, KNN Regressor.  
   - Visualizations include confusion matrices, accuracy barplots, correlation heatmaps, and feature importance charts (using Seaborn).

4. **Visualization**  
   - All model evaluation metrics and feature importance graphs are generated for each target variable using Seaborn.

## Usage
1. Clone this repository:
```bash
git clone <your-repo-url>
