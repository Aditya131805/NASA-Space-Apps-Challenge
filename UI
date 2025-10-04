import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Set modern theme
st.set_page_config(
    page_title="Planet Predictor 🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load trained models and scalers
def load_models(name):
    return pickle.load(open(f"{name}.pkl", "rb"))

# Radius models
Linear_Radius = load_models("linear_radius")
RF_Radius = load_models("rf_radius")
KNN_Radius = load_models("knn_radius")
Scaler_Radius = load_models("scaler_radius")

# Stellar Mag models
Linear_Mag = load_models("linear_mag")
RF_Mag = load_models("rf_mag")
KNN_Mag = load_models("knn_mag")
Scaler_Mag = load_models("scaler_mag")

# Stellar Radius models
Linear_StRad = load_models("linear_st_rad")
RF_StRad = load_models("rf_st_rad")
KNN_StRad = load_models("knn_st_rad")
Scaler_StRad = load_models("scaler_st_rad")

# Surface Gravity models
Linear_Logg = load_models("linear_logg")
RF_Logg = load_models("rf_logg")
KNN_Logg = load_models("knn_logg")
Scaler_Logg = load_models("scaler_logg")

# Effective Temperature models
Linear_Teff = load_models("linear_teff")
RF_Teff = load_models("rf_teff")
KNN_Teff = load_models("knn_teff")
Scaler_Teff = load_models("scaler_teff")

# Stellar Mass models
Linear_Mass = load_models("linear_mass")
RF_Mass = load_models("rf_mass")
KNN_Mass = load_models("knn_mass")
Scaler_Mass = load_models("scaler_mass")

# Title
st.title("🌌 Planet Predictor Dashboard")
st.markdown("""
Predict key planetary and stellar parameters interactively using Machine Learning models:
- Linear Regression
- Random Forest
- KNN
""")

# Sidebar inputs
st.sidebar.header("Input Planet/Star Features")
def get_input():
    data = {
        "equilibrium_temp": st.sidebar.slider("Equilibrium Temperature (K)", 100, 5000, 3000),
        "distance": st.sidebar.slider("Distance (AU)", 0.01, 1000.0, 1.0),
        "radius": st.sidebar.slider("Planet Radius (R⊕)", 0.1, 50.0, 1.0),
        "period": st.sidebar.slider("Orbital Period (days)", 0.1, 1000.0, 365.0),
        "stellar_teff": st.sidebar.slider("Stellar Teff (K)", 2000, 10000, 5778),
        "stellar_logg": st.sidebar.slider("Stellar logg", 0.0, 5.0, 4.44),
        "stellar_rad": st.sidebar.slider("Stellar Radius (R☉)", 0.1, 10.0, 1.0),
        "stellar_mag": st.sidebar.slider("Stellar Magnitude", -10.0, 20.0, 4.83),
        "insolation": st.sidebar.slider("Insolation (W/m²)", 0.0, 100000.0, 1361.0),
        "stellar_mass": st.sidebar.slider("Stellar Mass (M☉)", 0.1, 10.0, 1.0)
    }
    return pd.DataFrame(data, index=[0])

input_df = get_input()

# Helper functions
def predict_models(models, scaler, features):
    scaled = scaler.transform(input_df[features])
    predictions = {name: model.predict(scaled)[0] for name, model in models.items()}
    return predictions

def plot_r2(models, scores):
    fig, ax = plt.subplots()
    sns.barplot(x=models, y=scores, palette="Greens_d", ax=ax)
    ax.set_ylim(0,1)
    ax.set_ylabel("R² Score")
    st.pyplot(fig)

def plot_feature_importance(rf_model, feature_names):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x=rf_model.feature_importances_, y=feature_names, palette='Oranges_d', ax=ax)
    ax.set_title("Feature Importance - Random Forest")
    st.pyplot(fig)

# Tabs
tabs = st.tabs(["Radius", "Insolation", "Stellar Mag", "Stellar Radius", "Surface Gravity", "Effective Temp", "Stellar Mass"])

# Radius Tab
with tabs[0]:
    st.header("Planet Radius Prediction")
    features_radius = ['equilibrium_temp','distance','period','stellar_teff','stellar_logg','stellar_rad','stellar_mag','insolation','stellar_mass']
    models_radius = {
        "Linear Regression": Linear_Radius,
        "Random Forest": RF_Radius,
        "KNN": KNN_Radius
    }
    preds = predict_models(models_radius, Scaler_Radius, features_radius)
    col1, col2, col3 = st.columns(3)
    col1.metric("Linear Regression", round(preds["Linear Regression"], 3))
    col2.metric("Random Forest", round(preds["Random Forest"], 3))
    col3.metric("KNN", round(preds["KNN"], 3))
    st.subheader("Feature Importance - Random Forest")
    plot_feature_importance(RF_Radius, features_radius)

# Stellar Mag Tab
with tabs[2]:
    st.header("Stellar Magnitude Prediction")
    features_mag = ['equilibrium_temp','distance','radius','period','stellar_teff','stellar_logg','stellar_rad','insolation','stellar_mass']
    models_mag = {
        "Linear Regression": Linear_Mag,
        "Random Forest": RF_Mag,
        "KNN": KNN_Mag
    }
    preds = predict_models(models_mag, Scaler_Mag, features_mag)
    col1, col2, col3 = st.columns(3)
    col1.metric("Linear Regression", round(preds["Linear Regression"], 3))
    col2.metric("Random Forest", round(preds["Random Forest"], 3))
    col3.metric("KNN", round(preds["KNN"], 3))
    st.subheader("Feature Importance - Random Forest")
    plot_feature_importance(RF_Mag, features_mag)

# Stellar Radius Tab
with tabs[3]:
    st.header("Stellar Radius Prediction")
    features_strad = ['equilibrium_temp','distance','radius','period','stellar_teff','stellar_logg','stellar_mag','insolation','stellar_mass']
    models_strad = {
        "Linear Regression": Linear_StRad,
        "Random Forest": RF_StRad,
        "KNN": KNN_StRad
    }
    preds = predict_models(models_strad, Scaler_StRad, features_strad)
    col1, col2, col3 = st.columns(3)
    col1.metric("Linear Regression", round(preds["Linear Regression"], 3))
    col2.metric("Random Forest", round(preds["Random Forest"], 3))
    col3.metric("KNN", round(preds["KNN"], 3))
    st.subheader("Feature Importance - Random Forest")
    plot_feature_importance(RF_StRad, features_strad)

# Surface Gravity Tab
with tabs[4]:
    st.header("Surface Gravity Prediction")
    features_logg = ['equilibrium_temp','distance','radius','period','stellar_teff','stellar_rad','stellar_mag','insolation','stellar_mass']
    models_logg = {
        "Linear Regression": Linear_Logg,
        "Random Forest": RF_Logg,
        "KNN": KNN_Logg
    }
    preds = predict_models(models_logg, Scaler_Logg, features_logg)
    col1, col2, col3 = st.columns(3)
    col1.metric("Linear Regression", round(preds["Linear Regression"], 3))
    col2.metric("Random Forest", round(preds["Random Forest"], 3))
    col3.metric("KNN", round(preds["KNN"], 3))
    st.subheader("Feature Importance - Random Forest")
    plot_feature_importance(RF_Logg, features_logg)

# Effective Temperature Tab
with tabs[5]:
    st.header("Stellar Effective Temperature Prediction")
    features_teff = ['stellar_logg','equilibrium_temp','distance','radius','period','stellar_rad','stellar_mag','insolation','stellar_mass']
    models_teff = {
        "Linear Regression": Linear_Teff,
        "Random Forest": RF_Teff,
        "KNN": KNN_Teff
    }
    preds = predict_models(models_teff, Scaler_Teff, features_teff)
    col1, col2, col3 = st.columns(3)
    col1.metric("Linear Regression", round(preds["Linear Regression"], 3))
    col2.metric("Random Forest", round(preds["Random Forest"], 3))
    col3.metric("KNN", round(preds["KNN"], 3))
    st.subheader("Feature Importance - Random Forest")
    plot_feature_importance(RF_Teff, features_teff)

# Stellar Mass Tab
with tabs[6]:
    st.header("Stellar Mass Prediction")
    features_mass = ['stellar_logg','equilibrium_temp','distance','radius','period','stellar_teff','stellar_rad','stellar_mag','insolation']
    models_mass = {
        "Linear Regression": Linear_Mass,
        "Random Forest": RF_Mass,
        "KNN": KNN_Mass
    }
    preds = predict_models(models_mass, Scaler_Mass, features_mass)
    col1, col2, col3 = st.columns(3)
    col1.metric("Linear Regression", round(preds["Linear Regression"], 3))
    col2.metric("Random Forest", round(preds["Random Forest"], 3))
    col3.metric("KNN", round(preds["KNN"], 3))
    st.subheader("Feature Importance - Random Forest")
    plot_feature_importance(RF_Mass, features_mass)
