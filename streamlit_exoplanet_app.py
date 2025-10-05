import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import io
import pickle
import os
import joblib

model_files = {
    'disposition': {
        'Random Forest': 'models/random_forest_classifier.pkl',
        'KNN': 'models/knn_classifier.pkl',
        'Logistic Regression': 'models/logistic_regression.pkl'
    },
    'radius': {
        'Linear Regression': 'models/linear_regression1.pkl',
        'Random Forest': 'models/random_forest_regressor1.pkl',
        'KNN': 'models/knn_regressor1.pkl'
    },
    'distance': {
        'Linear Regression': 'models/linear_regression2.pkl',
        'Random Forest': 'models/random_forest_regressor2.pkl',
        'KNN': 'models/knn_regressor2.pkl'
    },
    'equilibrium_temp': {
        'Linear Regression': 'models/linear_regression3.pkl',
        'Random Forest': 'models/random_forest_regressor3.pkl',
        'KNN': 'models/knn_regressor3.pkl'
    },
    'insolation': {
        'Linear Regression': 'models/linear_regression4.pkl',
        'Random Forest': 'models/random_forest_regressor4.pkl',
        'KNN': 'models/knn_regressor4.pkl'
    },
    'stellar_mag': {
        'Linear Regression': 'models/linear_regression5.pkl',
        'Random Forest': 'models/random_forest_regressor5.pkl',
        'KNN': 'models/knn_regressor5.pkl'
    },
    'stellar_rad': {
        'Linear Regression': 'models/linear_regression6.pkl',
        'Random Forest': 'models/random_forest_regressor6.pkl',
        'KNN': 'models/knn_regressor6.pkl'
    },
    'stellar_logg': {
        'Linear Regression': 'models/linear_regression7.pkl',
        'Random Forest': 'models/random_forest_regressor7.pkl',
        'KNN': 'models/knn_regressor7.pkl'
    },
    'stellar_teff': {
        'Linear Regression': 'models/linear_regression8.pkl',
        'Random Forest': 'models/random_forest_regressor8.pkl',
        'KNN': 'models/knn_regressor8.pkl'
    },
    'stellar_mass': {
        'Linear Regression': 'models/linear_regression9.pkl',
        'Random Forest': 'models/random_forest_regressor9.pkl',
        'KNN': 'models/knn_regressor9.pkl'
    }
}

st.set_page_config(page_title="Exoplanet ML Explorer", layout='wide')
st.title("🚀 Exoplanet ML Explorer — NASA Space Apps Project")

REGRESSION_TARGETS = ['radius', 'distance', 'equilibrium_temp', 'insolation',
                      'stellar_mag', 'stellar_rad', 'stellar_logg', 'stellar_teff', 'stellar_mass']
CLASSIFICATION_TARGETS = ['disposition']

st.sidebar.header("Task")
task_mode = st.sidebar.selectbox("Select mode", ["Visualize", "Predict"])

st.sidebar.header("Data source")
use_default = st.sidebar.checkbox('Load Combined Dataset.csv if present', value=True)
uploaded_file = st.sidebar.file_uploader("Or upload a CSV file")

@st.cache_data
def load_data(buffer):
    return pd.read_csv(buffer)

data = None
if uploaded_file is not None:
    try:
        data = load_data(uploaded_file)
        st.sidebar.success('Loaded uploaded CSV')
    except:
        st.sidebar.error('Failed to load uploaded CSV')
elif use_default:
    try:
        data = pd.read_csv('Combined Dataset.csv')
        st.sidebar.success('Loaded Combined Dataset.csv')
    except:
        st.sidebar.info('Combined Dataset.csv not found')

if data is None:
    st.warning('No dataset loaded yet.')
    st.stop()

st.write('### Dataset preview')
st.dataframe(data.head())

if 'disposition' in data.columns:
    mapping = {'CONFIRMED': 1, 'CANDIDATE': 0, 'FALSE POSITIVE': 0, 'ACTIVE PLANET CANDIDATE': 0, 'FALSE ALARM': 0, 'REFUTED': 0}
    try:
        data['disposition'] = data['disposition'].replace(mapping)
    except:
        pass

if task_mode == "Visualize":
    mode = st.sidebar.selectbox('Task type', ['Regression', 'Classification'])
    if mode == 'Regression':
        target = st.sidebar.selectbox('Select regression target', REGRESSION_TARGETS)
    else:
        target = st.sidebar.selectbox('Select classification target', CLASSIFICATION_TARGETS)

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    default_features = [c for c in numeric_cols if c != target]

    st.sidebar.header('Features')
    selected_features = st.sidebar.multiselect('Choose features', default_features, default=default_features)
    if not selected_features:
        st.sidebar.error('Select at least one feature')
        st.stop()

    st.sidebar.header('Model')
    if mode == 'Regression':
        model_choice = st.sidebar.selectbox('Model', ['Linear Regression', 'Random Forest Regressor', 'KNN Regressor'])
    else:
        model_choice = st.sidebar.selectbox('Model', ['Logistic Regression', 'Random Forest Classifier', 'KNN Classifier'])

    test_size = st.sidebar.slider('Test set size (%)', 5, 50, 20)
    random_state = st.sidebar.number_input('Random seed', value=42)

    if st.sidebar.button('Train model'):
        st.write(f'### Training {model_choice} to predict `{target}`')
        df = data.copy()
        subset_cols = selected_features + [target]
        df = df.dropna(subset=subset_cols)
        X = df[selected_features].values
        y = df[target].values
        if mode == 'Classification':
            y = y.astype(int)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size/100.0, random_state=int(random_state))
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
        with st.spinner('Fitting model...'):
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
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
        if mode == 'Regression':
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test, y=y_pred, ax=ax)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'Actual vs Predicted — {target}')
            st.pyplot(fig)
        if mode == 'Classification':
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
        if 'Random Forest' in model_choice:
            try:
                importances = model.feature_importances_
                fi_df = pd.DataFrame({'feature': selected_features, 'importance': importances})
                fi_df = fi_df.sort_values('importance', ascending=True)
                fig, ax = plt.subplots(figsize=(6, max(3, len(selected_features)*0.4)))
                sns.barplot(x='importance', y='feature', data=fi_df, ax=ax)
                ax.set_title('Feature Importance')
                st.pyplot(fig)
            except:
                pass
        if st.sidebar.checkbox('Run 5-fold cross validation'):
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2' if mode=='Regression' else 'accuracy')
            st.write('Cross-validation scores:', cv_scores)
            st.write('CV mean:', np.mean(cv_scores))
        buffer = io.BytesIO()
        pkg = {'model': model, 'scaler': scaler, 'features': selected_features, 'target': target, 'mode': mode}
        pickle.dump(pkg, buffer)
        buffer.seek(0)
        st.download_button('Download trained model', data=buffer, file_name=f'model_{target}_{model_choice}.pkl')


else:
    st.sidebar.header("Prediction")
    target = st.sidebar.selectbox("Select Target to Predict", list(model_files.keys()))
    model_name = st.sidebar.selectbox("Select Model", list(model_files[target].keys()))
    model_file = model_files[target][model_name]

    if not os.path.exists(model_file):
        st.error(f"Model file not found: {model_file}")
        st.stop()

    try:
        loaded_obj = joblib.load(model_file)
        if isinstance(loaded_obj, dict) and 'model' in loaded_obj:
            model = loaded_obj['model']
            saved_features = loaded_obj.get('features', None)
        else:
            model = loaded_obj
            saved_features = getattr(model, 'feature_names_in_', None)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    st.subheader("Enter Feature Values:")
    feature_input = {}

    if target == 'disposition':
        features = ['equilibrium_temp','distance','radius','period',
                    'stellar_teff','stellar_logg','stellar_mag','insolation','stellar_mass']
    else:
        features = saved_features
        if features is None:
            features = [c for c in data.select_dtypes(include=[np.number]).columns if c != target]

        if target in REGRESSION_TARGETS and 'disposition' in features:
            features.remove('disposition')

    for feat in features:
        feature_input[feat] = st.number_input(f"{feat}", value=0.0, format="%.6f")

    if st.button("Predict"):
        X_input = pd.DataFrame([feature_input])
        try:
            prediction = model.predict(X_input)[0]
            st.success(f"Prediction for **{target}** using **{model_name}**: {prediction}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")


