import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="House Price AI", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    h1 {
        color: #00ffcc;
    }
    .stButton>button {
        background-color: #00ffcc;
        color: black;
        font-weight: bold;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🏠 AI House Price Predictor")
st.markdown("### Smart Real Estate Prediction System")

# ---------------- LOAD DATA ----------------
boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame

X = df.drop("MEDV", axis=1)
y = df["MEDV"]

X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.mean())

# ---------------- MODEL ----------------
@st.cache_resource
def train_model():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("🏡 Enter Property Details")

values = []

for col in X.columns:

    if col == "RM":
        val = st.sidebar.slider("🏠 Average Rooms", 1.0, 10.0, 5.0)

    elif col == "LSTAT":
        val = st.sidebar.slider("📉 Lower Status %", 0.0, 40.0, 10.0)

    elif col == "AGE":
        val = st.sidebar.slider("🏚️ Property Age", 0.0, 100.0, 50.0)

    else:
        val = st.sidebar.number_input(f"{col}", value=float(X[col].mean()))

    values.append(val)

# ---------------- PREDICTION ----------------
st.subheader("💰 Prediction Result")

if st.button("🚀 Predict Price"):
    input_df = pd.DataFrame([values], columns=X.columns)
    prediction = model.predict(input_df)

    st.markdown(f"""
        <div style="padding:20px; border-radius:15px; background-color:#1c1f26">
            <h2 style="color:#00ffcc;">Estimated Price</h2>
            <h1 style="color:white;">${prediction[0]*1000:,.2f}</h1>
        </div>
    """, unsafe_allow_html=True)

# ---------------- FEATURE IMPORTANCE ----------------
st.subheader("📊 Feature Importance")

if st.checkbox("Show Feature Importance"):
    importances = model.feature_importances_
    features = X.columns

    fig, ax = plt.subplots()
    ax.barh(features, importances)
    ax.set_title("Feature Importance")

    st.pyplot(fig)