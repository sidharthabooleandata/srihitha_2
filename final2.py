import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ===============================
# Load & preprocess data
# ===============================
FILE_PATH = "1Y02_Data.xlsx"

@st.cache_data
def load_data():
    try:
        if not os.path.exists(FILE_PATH):
            st.error(f"File not found: {FILE_PATH}")
            return pd.DataFrame()
        all_sheets = pd.read_excel(FILE_PATH, sheet_name=None)
        df = pd.concat(all_sheets.values(), ignore_index=True)

        # Drop useless columns
        drop_cols = ["Unnamed: 6", "Unnamed: 7", "Unnamed: 12",
                     "4AS3BKS", "recordstamp"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_resource
def get_processed_data_and_model():
    df = load_data()
    if df.empty:
        return None, None, None, None, None, None

    # Target & top features
    target = "BFOUT"

    if target not in df.columns:
        st.error(f"Target column {target} not found")
        return None, None, None, None, None, None

    df = df.dropna(subset=[target])
    if df.empty:
        st.error("No data after removing null targets")
        return None, None, None, None, None, None

    X = df.drop(columns=[target])
    y = df[target]

    # Handle datetime columns
    for col in X.select_dtypes(include=["datetime64[ns]"]).columns:
        try:
            X[col + "_year"] = X[col].dt.year
            X[col + "_month"] = X[col].dt.month
            X[col + "_day"] = X[col].dt.day
            X[col + "_dayofweek"] = X[col].dt.dayofweek
            X.drop(columns=[col], inplace=True)
        except Exception:
            X.drop(columns=[col], inplace=True)

    # Encode categorical columns
    label_encoders = {}
    for col in X.select_dtypes(include="object").columns:
        try:
            le = LabelEncoder()
            X[col] = X[col].fillna('unknown')
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        except Exception:
            X.drop(columns=[col], inplace=True)

    # Fill numeric NaN with mean
    X = X.fillna(X.mean())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    return df, X, y, model, label_encoders, target

# Get cached data and model
result = get_processed_data_and_model()
if result[0] is None:
    st.stop()

df, X, y, model, label_encoders, target = result

# Target & top features (BFIN removed)
top_features = [
    "TALLYLENGTH", "TALLYWIDTH", "MATERIAL",
    "MATERIALTHICKNESS", "MATERIALSPECIE", "TALLYGRADE"
]

# ===============================
# Streamlit User Input (ONLY top features)
# ===============================
st.subheader("🔮 Predict BFOUT")

input_data = {}
for col in top_features:
    if col in X.columns:
        if col in label_encoders:
            options = list(label_encoders[col].classes_)
            input_val = st.selectbox(f"Select {col}", options)
            try:
                input_data[col] = label_encoders[col].transform([input_val])[0]
            except Exception:
                input_data[col] = 0
        else:
            try:
                input_val = st.number_input(
                    f"Enter {col}",
                    float(X[col].min()),
                    float(X[col].max()),
                    float(X[col].mean())
                )
                input_data[col] = input_val
            except Exception:
                input_val = st.number_input(f"Enter {col}", value=0.0)
                input_data[col] = input_val

# Fill missing columns with defaults
for col in X.columns:
    if col not in input_data:
        if col in label_encoders:
            input_data[col] = 0  # default category
        else:
            try:
                input_data[col] = float(X[col].mean())
            except Exception:
                input_data[col] = 0.0

# ✅ Align order with training features
input_df = pd.DataFrame([input_data])[X.columns]

if st.button("Predict BFOUT"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🎯 Predicted BFOUT: {prediction:.2f}")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")


