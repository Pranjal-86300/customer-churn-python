# Customer Churn Prediction App

A machine learning web application built with **Streamlit** to predict whether a telecom customer will **churn** or continue their subscription.  
The model is trained using **Logistic Regression**, with preprocessing handled using a Scikit-learn **Pipeline** and **OneHotEncoder** to ensure consistent predictions.

---

## 🔍 Project Overview

Customer churn prediction is critical for subscription-based businesses.  
This application predicts:

- Whether a customer is likely to churn  
- Probability of Churn (0–100%)  
- Probability of Not Churning (0–100%)

The app uses customer attributes such as:

- Demographics (Gender, Senior Citizen)
- Subscription details (Contract, Internet Service)
- Billing information (Payment Method, Monthly Charges)
- Services used (Online Security, Streaming, Backup, etc.)
- Tenure

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Streamlit** – Web UI
- **Scikit-learn** – ML model + preprocessing
- **Pandas, NumPy** – Data processing
- **Joblib** – Model saving/loading

---

## 📁 Folder Structure

```
my-churn-app/
│
├── app.py                 # Streamlit web app
├── model.joblib           # Saved ML model (Logistic Regression Pipeline)
├── churn.csv              # Dataset used for UI dropdown options
├── Churn.ipynb            # Model training notebook
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 📚 Model Training Summary

Model was trained in **Churn.ipynb** with the following workflow:

1. Load dataset (`churn.csv`)
2. Drop `customerID`
3. Convert target `Churn` → numeric (Yes=1, No=0)
4. Split into train/test sets
5. Build a preprocessing + training pipeline:
   - OneHotEncoder for categorical features  
   - Passthrough for numeric features  
6. Train a Logistic Regression classifier  
7. Save trained model as `model.joblib`

Training template:

```python
pipeline = Pipeline([
    ("preprocess", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("num", "passthrough", numeric_columns)
    ])),
    ("clf", LogisticRegression(max_iter=500))
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "model.joblib")
```

---

## ▶️ Running the App Locally

**1. Install dependencies**

```
pip install -r requirements.txt
```

**2. Run the Streamlit App**

```
streamlit run app.py
```

App will open at:

```
http://localhost:8501
```

---

## ☁️ Deploy on Streamlit Cloud (Recommended)

1. Push this project to a GitHub repository  
2. Visit: https://share.streamlit.io  
3. Sign in with GitHub  
4. Click **New App**  
5. Select:
   - Repository: `your-username/my-churn-app`
   - Branch: `main`
   - Main file: `app.py`
6. Click **Deploy**

Streamlit Cloud will automatically:

- Install dependencies  
- Launch the application  
- Give you a public URL  

---

## 📦 requirements.txt

```
streamlit
pandas
numpy
scikit-learn
joblib
```

---

## 👨‍💻 Author

(Add your details here)

---

# End of README
