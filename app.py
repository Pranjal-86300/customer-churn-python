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

# Load dataset (only for UI options)
df = pd.read_csv("churn.csv")

# Drop ID + target if present
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)
if "Churn" in df.columns:
    df = df.drop("Churn", axis=1)

# CLEANING EXACTLY LIKE TRAINING --------------------------------------

service_cols = [
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

# Clean Yes/No style columns (Internet-service related)
for col in service_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "yes": "Yes",
            "no": "No",
            "no internet service": "No"
        })
    )

# Clean MultipleLines
df["MultipleLines"] = (
    df["MultipleLines"]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace({
        "yes": "Yes",
        "no": "No",
        "no phone service": "No"
    })
)

# Standardize to Title Case
for col in service_cols + ["MultipleLines"]:
    df[col] = df[col].str.title()

# NOW df_ui is safe for UI dropdowns
df_ui = df.copy()


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

# Title
st.markdown(
    """
    <h1 style='text-align:center;'>🧠 Customer Churn Prediction</h1>
    <p style='text-align:center; color:#bbb; font-size:18px;'>
        Fill in customer details below to predict churn probability.
    </p>
    """,
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------------
# Input Features
# -----------------------------------
FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

# Use df_ui as reference for unique values
df_ui = df_ui.reindex(columns=FEATURES)  # ensure order, will insert NaN if missing

st.markdown("<div class='stCard'>", unsafe_allow_html=True)
st.subheader("📌 Enter Customer Information")

cols = st.columns(3)
user_input = {}
i = 0

for col in FEATURES:
    with cols[i]:
        # Special handling for SeniorCitizen, tenure and TotalCharges
        if col == "SeniorCitizen":
            # Show Yes/No and convert later to 1/0
            user_input[col] = st.selectbox("Senior Citizen", ["No", "Yes"])
        elif col == "tenure":
            # integer input for tenure (months)
            default_tenure =  int(df_ui[col].dropna().median()) if col in df_ui else 1
            user_input[col] = st.number_input(
                "tenure (months)",
                min_value=0,
                max_value=1000,
                value=int(default_tenure),
                step=1,
                format="%d"
            )
        elif col == "TotalCharges":
            # float input for total charges
            # often stored as object if missing; we provide float input
            # default to mean if numeric
            default_total = None
            if col in df_ui:
                try:
                    default_total = float(pd.to_numeric(df_ui[col], errors="coerce").dropna().mean())
                except Exception:
                    default_total = 0.0
            if default_total is None or np.isnan(default_total):
                default_total = 0.0
            user_input[col] = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=1_000_000.0,
                value=float(round(default_total, 2)),
                step=0.1,
                format="%.2f"
            )
        else:
            # For other columns, choose selectbox for categorical, number_input for numeric
            if col in df_ui and df_ui[col].dtype == object:
                opts = sorted(df_ui[col].dropna().unique().tolist())
                if len(opts) == 0:
                    # fallback if column exists but no unique values found
                    user_input[col] = st.text_input(col, value="")
                else:
                    user_input[col] = st.selectbox(col, opts)
            else:
                # numeric columns: MonthlyCharges (float), etc.
                default_val = None
                if col in df_ui:
                    default_val = df_ui[col].dropna().mean()
                if default_val is None or np.isnan(default_val):
                    default_val = 0.0
                # For MonthlyCharges we want float with decimals
                if col == "MonthlyCharges":
                    user_input[col] = st.number_input(
                        col,
                        min_value=0.0,
                        max_value=10000.0,
                        value=float(round(default_val, 2)),
                        step=0.1,
                        format="%.2f"
                    )
                else:
                    # generic numeric fallback (use float)
                    user_input[col] = st.number_input(
                        col,
                        value=float(default_val) if default_val is not None else 0.0,
                        step=0.1
                    )
    i = (i + 1) % 3

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------
# Post-process user_input to match training types
# -----------------------------------
# Convert SeniorCitizen Yes/No -> 1 / 0 (int)
if "SeniorCitizen" in user_input:
    sc = user_input["SeniorCitizen"]
    if isinstance(sc, str):
        sc_mapped = 1 if sc.strip().lower() in ["yes", "1", "true", "y"] else 0
    else:
        # if already numeric
        sc_mapped = int(sc)
    user_input["SeniorCitizen"] = sc_mapped

# Ensure tenure is integer
if "tenure" in user_input:
    try:
        user_input["tenure"] = int(user_input["tenure"])
    except Exception:
        user_input["tenure"] = int(float(user_input["tenure"]))

# Ensure MonthlyCharges and TotalCharges are floats
for fld in ["MonthlyCharges", "TotalCharges"]:
    if fld in user_input:
        try:
            user_input[fld] = float(user_input[fld])
        except Exception:
            user_input[fld] = 0.0

# -----------------------------------
# Predict Button + Enhanced Probability Output
# -----------------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Predict Churn", use_container_width=True):

    # Build DataFrame for model input
    input_df = pd.DataFrame([user_input], columns=FEATURES)

    # Pipeline handles preprocessing automatically (OneHotEncoder etc.)
    try:
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
    except Exception as e:
        st.error("Model prediction failed. See error details below:")
        st.exception(e)
    else:
        churn_prob = proba[1] * 100
        no_churn_prob = proba[0] * 100

        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.subheader("📊 Prediction Result")

        if pred == 1:
            st.markdown(
                f"""
                <h2 style='color:#ff4b4b;'>⚠️ Customer Likely to Churn</h2>
                <p style='font-size:22px;'>
                    <b>Churn Probability:</b> {churn_prob:.2f}%<br>
                    <b>Not Churn Probability:</b> {no_churn_prob:.2f}%
                </p>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <h2 style='color:#4CAF50;'>✅ Customer Not Likely to Churn</h2>
                <p style='font-size:22px;'>
                    <b>Not Churn Probability:</b> {no_churn_prob:.2f}%<br>
                    <b>Churn Probability:</b> {churn_prob:.2f}%
                </p>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

