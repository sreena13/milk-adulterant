import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load dataset
try:
    df = pd.read_csv("Adulterant-dataset.csv")
except FileNotFoundError:
    st.error("Dataset not found! Please check the file path.")
    st.stop()

X = df[['ml', 'F', 'D', 'L', 'S', 'P', 'W']]
y = df['target']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("Adulterant Detection System")
st.write("Enter the values to predict the type of adulterant:")

# Input fields
ml = st.number_input("ml", value=0.027, format="%.3f")
F = st.number_input("F", value=6.6, format="%.2f")
D = st.number_input("D", value=19.89, format="%.2f")
L = st.number_input("L", value=3.06, format="%.2f")
S = st.number_input("S", value=6.73, format="%.2f")
P = st.number_input("P", value=2.37, format="%.2f")
W = st.number_input("W", value=0.0, format="%.2f")

# Prediction function
def map_condition(pred):
    if 0 <= pred <= 1.5:
        return "Starch"
    elif 1.6 <= pred <= 2.5:
        return "Sucrose"
    elif 2.6 <= pred <= 3.5:
        return "NaNO3"
    elif 3.6 <= pred <= 4.5:
        return "Urea"
    elif 4.6 <= pred <= 5.5:
        return "Glucose"
    elif 5.6 <= pred <= 6.5:
        return "NaCl"
    elif 6.6 <= pred <= 7.5:
        return "Formaldehyde"
    else:
        return "Unknown"

if st.button("Predict"):
    input_data = pd.DataFrame([[ml, F, D, L, S, P, W]], columns=['ml', 'F', 'D', 'L', 'S', 'P', 'W'])
    pred_value = model.predict(input_data)[0]
    condition = map_condition(pred_value)
    st.success(f"Predicted Value: {pred_value:.2f}")
    st.info(f"Condition: {condition}")
