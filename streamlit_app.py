import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt

# Load the dataset and model
df = pd.read_csv("Data Sets\\Adulterant-dataset.csv")
X = df[['ml', 'F', 'D', 'L', 'S', 'P', 'W']]
y = df['target']

# Train the LassoCV model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso_cv_model = LassoCV(cv=2)
lasso_cv_model.fit(X_scaled, y)

# Streamlit user interface
st.title("Milk Adulterant Prediction")

# Input features from the user
ml = st.number_input("ml", min_value=0.0)
F = st.number_input("F", min_value=0.0)
D = st.number_input("D", min_value=0.0)
L = st.number_input("L", min_value=0.0)
S = st.number_input("S", min_value=0.0)
P = st.number_input("P", min_value=0.0)
W = st.number_input("W", min_value=0.0)

# Predict the output
user_input = [[ml, F, D, L, S, P, W]]
user_input_scaled = scaler.transform(user_input)  # Scale the user input
predicted_value = lasso_cv_model.predict(user_input_scaled)

# Show the predicted value
st.write(f"Predicted Value: {predicted_value[0]}")

# Plotting (Optional)
if st.button('Plot True vs Predicted'):
    y_pred_test = lasso_cv_model.predict(X_scaled)
    
    plt.scatter(y, y_pred_test, color='blue')
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    st.pyplot(plt)
