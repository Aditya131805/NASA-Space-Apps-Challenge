# Streamlit Exoplanet Regression & Classification Explorer
# Save this file as streamlit_exoplanet_app.py and run: streamlit run streamlit_exoplanet_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             r2_score, mean_squared_error, mean_absolute_error)
import matplotlib.pyplot as plt
import seaborn as sns
import io
import pickle

st.set_page_config(page_title="Exoplanet ML Explorer", layout='wide')
st.title("🚀 Exoplanet ML Explorer — NASA Space Apps Project")
st.write("✅ Streamlit app started successfully — UI is loading...")


# DEFAULT TARGETS (from your project)
REGRESSION_TARGETS = [
    'radius', 'distance', 'equilibrium_temp', 'insolation',
    'stellar_mag', 'stellar_rad', 'stellar_logg', 'stellar_teff', 'stellar_mass'
]
CLASSIFICATION_TARGETS = ['disposition']
ALL_TARGETS = REGRESSION_TARGETS + CLASSIFICATION_TARGETS

st.sidebar.header("Data source")
use_default = st.sidebar.checkbox('Load Combined Dataset.csv if present (recommended)', value=True)
uploaded_file = st.sidebar.file_uploader("Or upload a CSV file (cleaned individual or combined)")

@st.cache_data
def load_data_from_buffer(buffer):
    return pd.read_csv(buffer)

# Load data logic
data = None
if uploaded_file is not None:
    try:
        data = load_data_from_buffer(uploaded_file)
        st.sidebar.success('Loaded uploaded CSV')
    except Exception as e:
        st.sidebar.error(f'Failed to load uploaded CSV: {e}')
elif use_default:
    try:
        data = pd.read_csv('Combined Dataset.csv')
        st.sidebar.success('Loaded Combined Dataset.csv from working directory')
    except Exception:
        st.sidebar.info('Combined Dataset.csv not found in working directory. You can upload a CSV.')

if data is None:
    st.warning('No dataset loaded yet. Upload a cleaned CSV or place Combined Dataset.csv in the app folder.')
    st.stop()

st.write('### Dataset preview')
st.dataframe(data.head())

# Basic cleaning: ensure disposition encoded if exists
if 'disposition' in data.columns:
    # map common text dispositions to ints where possible
    mapping = {
        'CONFIRMED': 1,
        'CANDIDATE': 0,
        'FALSE POSITIVE': 0,
        'ACTIVE PLANET CANDIDATE': 0,
        'FALSE ALARM': 0,
        'REFUTED': 0
    }
    try:
        data['disposition'] = data['disposition'].replace(mapping)
    except Exception:
        pass

# User selects target
st.sidebar.header('Model & Target Selection')
mode = st.sidebar.selectbox('Task type', ['Regression', 'Classification'])
if mode == 'Regression':
    target = st.sidebar.selectbox('Select regression target', REGRESSION_TARGETS)
else:
    target = st.sidebar.selectbox('Select classification target', CLASSIFICATION_TARGETS)

# Feature selection
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if target in numeric_cols:
    default_features = [c for c in numeric_cols if c != target]
else:
    default_features = numeric_cols

st.sidebar.header('Features')
selected_features = st.sidebar.multiselect('Choose features (at least 1)', default_features, default=default_features)
if not selected_features:
    st.sidebar.error('Select at least one feature to train the model')
    st.stop()

# Model selection
st.sidebar.header('Model')
if mode == 'Regression':
    model_choice = st.sidebar.selectbox('Model', ['Linear Regression', 'Random Forest Regressor', 'KNN Regressor'])
else:
    model_choice = st.sidebar.selectbox('Model', ['Logistic Regression', 'Random Forest Classifier', 'KNN Classifier'])

# Train-test split and random seed
test_size = st.sidebar.slider('Test set size (%)', 5, 50, 20)
random_state = st.sidebar.number_input('Random seed', value=42)

# Training button
if st.sidebar.button('Train model'):
    st.write(f'### Training {model_choice} to predict `{target}`')

    # Prepare X, y
    df = data.copy()
    # Drop rows with NA in selected cols or target
    subset_cols = selected_features + [target]
    df = df.dropna(subset=subset_cols)

    X = df[selected_features].values
    y = df[target].values

    # For classification, ensure y is integer labels
    if mode == 'Classification':
        y = y.astype(int)

    # Scale features for some models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size/100.0, random_state=int(random_state))

    # Initialize model
    if mode == 'Regression':
        if model_choice == 'Linear Regression':
            model = LinearRegression()
        elif model_choice == 'Random Forest Regressor':
            model = RandomForestRegressor(n_estimators=100, random_state=int(random_state))
        else:
            model = KNeighborsRegressor(n_neighbors=5)
    else:
        if model_choice == 'Logistic Regression':
            model = LogisticRegression(max_iter=2000)
        elif model_choice == 'Random Forest Classifier':
            model = RandomForestClassifier(n_estimators=100, random_state=int(random_state))
        else:
            model = KNeighborsClassifier(n_neighbors=5)

    # Fit
    with st.spinner('Fitting model...'):
        model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics and display
    col1, col2 = st.columns(2)
    with col1:
        if mode == 'Regression':
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            st.metric('R² Score', f'{r2:.4f}')
            st.metric('MSE', f'{mse:.4f}')
            st.metric('MAE', f'{mae:.4f}')
        else:
            acc = accuracy_score(y_test, y_pred)
            st.metric('Accuracy', f'{acc:.4f}')
            st.text('Classification Report:')
            st.text(classification_report(y_test, y_pred))

    with col2:
        st.write('### Sample Predictions (first 10)')
        sample_df = pd.DataFrame(X_test, columns=selected_features).copy()
        sample_df['actual'] = y_test
        sample_df['predicted'] = y_pred
        st.dataframe(sample_df.head(10))

    # Plot Actual vs Predicted for regression
    if mode == 'Regression':
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'Actual vs Predicted — {target}')
        st.pyplot(fig)

    # Confusion matrix for classification
    if mode == 'Classification':
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

    # Feature importance (for tree-based models)
    if 'Random Forest' in model_choice:
        try:
            importances = model.feature_importances_
            fi_df = pd.DataFrame({'feature': selected_features, 'importance': importances})
            fi_df = fi_df.sort_values('importance', ascending=True)
            fig, ax = plt.subplots(figsize=(6, max(3, len(selected_features)*0.4)))
            sns.barplot(x='importance', y='feature', data=fi_df, ax=ax)
            ax.set_title('Feature Importance')
            st.pyplot(fig)
        except Exception as e:
            st.write('Could not compute feature importances:', e)

    # Cross-validation (quick)
    if st.sidebar.checkbox('Run 5-fold cross validation (may take time)'):
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2' if mode=='Regression' else 'accuracy')
        st.write('Cross-validation scores:', cv_scores)
        st.write('CV mean:', np.mean(cv_scores))

    # Option to download the trained model and scaler
    buffer = io.BytesIO()
    pkg = {'model': model, 'scaler': scaler, 'features': selected_features, 'target': target, 'mode': mode}
    pickle.dump(pkg, buffer)
    buffer.seek(0)
    st.download_button('Download trained model (pickle)', data=buffer, file_name=f'model_{target}_{model_choice}.pkl')

st.write('---')
st.write('App notes:')
st.write('- This app defaults to `Combined Dataset.csv` if present in the working directory. You can also upload individual cleaned CSVs.')
st.write('- It supports all regression targets and the classification target `disposition`. If you want more classification labels, ensure `disposition` is encoded properly in your CSV.')
st.write('- For stellar_mass prediction issues: try log-transforming the target or cleaning outliers before uploading the CSV.')
