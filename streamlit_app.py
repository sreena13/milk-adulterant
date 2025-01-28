import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("Data Sets\\Adulterant-dataset.csv")

X = df[['ml', 'F', 'D', 'L', 'S', 'P', 'W']]
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on a custom input
input_data = pd.DataFrame([[0.027, 6.6, 19.89, 3.06, 6.73, 2.37, 0]], columns=['ml', 'F', 'D', 'L', 'S', 'P', 'W'])
pred_value = model.predict(input_data)[0]

# Function to map prediction ranges to conditions
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

# Map the predicted value
condition = map_condition(pred_value)

# Output the prediction and condition
print(f"Predicted Value: {pred_value}")
print(f"Condition: {condition}")
