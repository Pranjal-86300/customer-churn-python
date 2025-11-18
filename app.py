import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------------
# Load Model
# -----------------------------------
model = joblib.load("model.joblib")

# Load dataset (only to extract unique values)
df = pd.read_csv("churn.csv")
df = df.drop("customerID", axis=1)  # remove unnecessary ID


# -----------------------------------
# CSS for Beautiful UI
# -----------------------------------
st.markdown("""
<style>
body { background-color: #0d0d0d; }
.stCard {
    background-color: #1a1a1a;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #333;
    box-shadow: 0px 0px 10px rgba(255,255,255,0.05);
}
h1, h2, h3, p, label, .stTextInput, .stNumberInput {
    color: #f0f0f0 !important;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------
# Title Section
# -----------------------------------
st.markdown(
    """
    <h1 style='text-align:center;'>🧠 Customer Churn Prediction</h1>
    <p style='text-align:center; color:#bbb; font-size:18px;'>
        Fill in customer details below to predict if they are likely to churn.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("<hr>", unsafe_allow_html=True)


# -----------------------------------
# Input Features (same as training)
# -----------------------------------
FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

df = df[FEATURES]

# -----------------------------------
# Input Form (3-column layout)
# -----------------------------------
st.markdown("<div class='stCard'>", unsafe_allow_html=True)
st.subheader("📌 Enter Customer Information")

cols = st.columns(3)
user_input = {}

col_idx = 0

for col in FEATURES:
    with cols[col_idx]:
        if df[col].dtype == object:
            user_input[col] = st.selectbox(col, sorted(df[col].unique()))
        else:
            user_input[col] = st.number_input(
                col,
                value=float(df[col].mean()),
                step=1.0 if col in ["tenure"] else 0.1
            )
    col_idx = (col_idx + 1) % 3

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------
# Predict Button
# -----------------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Predict Churn", use_container_width=True):

    # Create DataFrame for prediction
    input_df = pd.DataFrame([user_input])

    # Pipeline handles preprocessing automatically
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    churn_prob = proba[1] * 100        # churn probability
    no_churn_prob = proba[0] * 100     # non-churn probability

    st.markdown("<div class='stCard'>", unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    if pred == 1:
        st.markdown(
            f"""
            <h2 style='color:#ff4b4b;'>⚠️ Customer Likely to Churn</h2>
            <p style='font-size:22px;'>
                <b>Churn Probability:</b> {churn_prob:.2f}%<br>
                <b>Not-Churn Probability:</b> {no_churn_prob:.2f}%
            </p>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <h2 style='color:#4CAF50;'>✅ Customer Not Likely to Churn</h2>
            <p style='font-size:22px;'>
                <b>Not-Churn Probability:</b> {no_churn_prob:.2f}%<br>
                <b>Churn Probability:</b> {churn_prob:.2f}%
            </p>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)
