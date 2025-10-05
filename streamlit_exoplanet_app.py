import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import requests
import pickle
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix, classification_report

import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie


st.set_page_config(page_title="Exoplanet ML Explorer", layout="wide", initial_sidebar_state="expanded")


PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
* {font-family: 'Poppins', sans-serif; transition: all 0.3s ease;}

/* Body background */
body { background-color: #0b1220; color: #e6eef8; }
.stApp { background: linear-gradient(180deg,#071029 0%, #071b2e 100%); }

/* Glassmorphic cards */
.card { background: rgba(255,255,255,0.05); border-radius:12px; padding:14px; margin-bottom:12px; box-shadow:0 6px 20px rgba(2,6,23,0.6); }
.card:hover { transform: translateY(-4px); transition: .18s ease; }

/* Header text */
.header { text-align:center; margin-bottom:8px; }
.h1 { font-size:34px; color:#7fd3ff; font-weight:700; letter-spacing:1px; }
.h2 { font-size:14px; color:#bcdcff; margin-top:-6px; }

/* Metrics */
.metric-big { font-size:28px; color:#fff; font-weight:700; }
.metric-sub { font-size:12px; color:#bcdcff; }

/* Prediction card */
.pred-card { background: linear-gradient(90deg,#744df6,#1fb6ff); color:white; padding:18px; border-radius:12px; text-align:center; }

/* Inputs */
.stMultiSelect, .stSelectbox, .stNumberInput { border-radius:8px; }

/* Footer */
.footer { color:#93b0c8; font-size:12px; text-align:center; margin-top:16px; }

/* Starfield */
#stars { position: fixed; width:100%; height:100%; background:black; z-index:-1; overflow:hidden; }
.star { position:absolute; width:2px; height:2px; background:white; border-radius:50%; opacity:0.8; animation: twinkle 5s infinite; }
@keyframes twinkle { 0%,100%{opacity:0.3;} 50%{opacity:1;} }

/* Light mode overrides */
body.light-mode .card { background: rgba(255,255,255,0.6); color:black; box-shadow: 0 6px 20px rgba(0,0,0,0.2); }
body.light-mode .h1 { color:#333; }
body.light-mode .h2 { color:#555; }
</style>

<div id="stars"></div>
<script>
for(let i=0;i<100;i++){
    let star=document.createElement('div');
    star.className='star';
    star.style.top=Math.random()*100+'%';
    star.style.left=Math.random()*100+'%';
    star.style.width=(Math.random()*2+1)+'px';
    star.style.height=(Math.random()*2+1)+'px';
    star.style.animationDuration=(Math.random()*5+2)+'s';
    document.getElementById('stars').appendChild(star);
}
</script>
"""
st.markdown(PAGE_CSS, unsafe_allow_html=True)


with st.sidebar:
    theme = st.radio("🎨 Theme", ["Dark Mode", "Light Mode"])
    if theme == "Light Mode":
        st.markdown("<body class='light-mode'></body>", unsafe_allow_html=True)

    st.markdown("<div class='card'><b>📁 Data</b></div>", unsafe_allow_html=True)
    use_default = st.checkbox("Load Combined Dataset.csv if present", value=True)
    uploaded_file = st.file_uploader("Or upload a CSV", type=["csv"])
    st.markdown("---")
    st.markdown("<div class='card'><b>⚙️ Training Options</b></div>", unsafe_allow_html=True)
    default_test_size = st.slider("Test Size (%)", 5, 50, 20)
    default_random_state = st.number_input("Random Seed", value=42)
    run_cv_default = st.checkbox("Enable 5-fold CV", value=False)


@st.cache_data
def read_csv_buffer(buf):
    return pd.read_csv(buf)

def safe_load_data():
    df = None
    if uploaded_file is not None:
        try: df = read_csv_buffer(uploaded_file)
        except: st.sidebar.error("Failed to load uploaded CSV.")
    elif use_default and os.path.exists("Combined Dataset.csv"):
        try: df = pd.read_csv("Combined Dataset.csv")
        except: st.sidebar.error("Failed to load Combined Dataset.csv")
    return df

def build_model(kind, model_choice, random_state):
    if kind=="Regression":
        if model_choice=="Linear Regression": return LinearRegression()
        if model_choice=="Random Forest": return RandomForestRegressor(n_estimators=120, random_state=random_state)
        return KNeighborsRegressor(n_neighbors=5)
    else:
        if model_choice=="Logistic Regression": return LogisticRegression(max_iter=2000)
        if model_choice=="Random Forest": return RandomForestClassifier(n_estimators=120, random_state=random_state)
        return KNeighborsClassifier(n_neighbors=5)

data = safe_load_data()
if data is None:
    st.warning("No dataset loaded yet. Upload CSV or place `Combined Dataset.csv` in folder.")
    st.stop()

if 'disposition' in data.columns:
    mapping = {'CONFIRMED':1,'CANDIDATE':0,'FALSE POSITIVE':0,'ACTIVE PLANET CANDIDATE':0,'FALSE ALARM':0,'REFUTED':0}
    data['disposition'] = data['disposition'].replace(mapping)


col_lottie, col_title = st.columns([1,4])
def load_lottie_url(url):
    try:
        r = requests.get(url, timeout=4)
        if r.status_code==200: return r.json()
    except: return None
lottie = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_x62chJ.json")
with col_lottie:
    if lottie: st_lottie(lottie, height=140)
with col_title:
    st.markdown("<div class='header'><div class='h1'>🚀 Exoplanet ML Explorer</div><div class='h2'>NASA Space Apps — Interactive ML Dashboard</div></div>", unsafe_allow_html=True)


tab1, tab2 = st.tabs(["📊 Visualize & Train", "🔮 Predict"])


with tab1:
    left, right = st.columns([2,1])
    with left: st.dataframe(data.head(80), use_container_width=True)
    with right:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        st.markdown(f"<div class='card'><b>Rows</b><div class='metric-big'>{data.shape[0]}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card'><b>Numeric cols</b><div class='metric-big'>{len(numeric_cols)}</div></div>", unsafe_allow_html=True)

    with st.expander("🔧 Configure Model & Features", expanded=True):
        model_kind = st.radio("Task", options=["Regression","Classification"], horizontal=True)
        if model_kind=="Regression":
            target = st.selectbox("Regression target", numeric_cols, index=0)
            model_choice = st.selectbox("Model", ["Linear Regression","Random Forest","KNN"])
        else:
            target = "disposition" if "disposition" in data.columns else st.selectbox("Classification target", numeric_cols)
            model_choice = st.selectbox("Model", ["Logistic Regression","Random Forest","KNN"])
        features = st.multiselect("Features", [c for c in numeric_cols if c!=target], default=[c for c in numeric_cols if c!=target][:6])
        test_size = st.slider("Test size (%)",5,50,default_test_size)
        random_state = st.number_input("Random seed", value=int(default_random_state))
        run_cv = st.checkbox("Run 5-fold CV", value=run_cv_default)

    if st.button("🚀 Train Model"):
        if not features: st.error("Select at least one feature.")
        else:
            df_clean = data.dropna(subset=features+[target])
            X = df_clean[features].values
            y = df_clean[target].values
            if model_kind=="Classification": y=y.astype(int)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=test_size/100.0, random_state=int(random_state))
            model = build_model(model_kind, model_choice, random_state=int(random_state))

            progress = st.progress(0)
            for p in range(3): time.sleep(0.25); progress.progress((p+1)*20)
            model.fit(X_train,y_train)
            progress.progress(100)
            st.success("Model trained!")

            y_pred = model.predict(X_test)
            left_metrics, right_plots = st.columns([1,2])
            with left_metrics:
                st.markdown("<div class='card'><b>Metrics</b></div>", unsafe_allow_html=True)
                if model_kind=="Regression":
                    st.markdown(f"<div class='card'><div class='metric-big'>{r2_score(y_test,y_pred):.4f}</div><div class='metric-sub'>R²</div></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='card'><div class='metric-big'>{accuracy_score(y_test,y_pred):.4f}</div><div class='metric-sub'>Accuracy</div></div>", unsafe_allow_html=True)
                if run_cv:
                    scores = cross_val_score(model,X_scaled,y,cv=5,scoring='r2' if model_kind=="Regression" else 'accuracy')
                    st.write("CV mean:", np.round(np.mean(scores),4))
            with right_plots:
                if model_kind=="Regression":
                    fig = px.scatter(x=y_test,y=y_pred,labels={'x':'Actual','y':'Predicted'}, trendline="ols")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.imshow(confusion_matrix(y_test,y_pred), text_auto=True)
                    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🔮 Predict")
    model_file = st.file_uploader("Upload trained model (.pkl)", type=["pkl"])
    if model_file:
        loaded = pickle.load(model_file)
        model = loaded.get("model")
        features = loaded.get("features")
        scaler = loaded.get("scaler")
        target_name = loaded.get("target","target")
        inputs = {}
        cols = st.columns(3)
        for i, feat in enumerate(features):
            col = cols[i%3]; inputs[feat]=col.number_input(feat, value=0.0)
        if st.button("🔭 Predict"):
            X_in = pd.DataFrame([inputs])
            X_proc = scaler.transform(X_in) if scaler else X_in.values
            pred = model.predict(X_proc)[0]
            st.markdown(f"<div class='pred-card'><div style='font-size:46px'>{np.round(pred,4)}</div></div>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🚀 NASA Space Apps Challenge</h1>", unsafe_allow_html=True)
