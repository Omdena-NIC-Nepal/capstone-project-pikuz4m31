import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("../data/processed/dailyclimate_cleaned.csv")

# Ensure 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Check for any rows with invalid Date values
if df['Date'].isnull().sum() > 0:
    st.warning(f"There are {df['Date'].isnull().sum()} invalid Date entries. These will be dropped.")
    df = df.dropna(subset=['Date'])

# Feature Engineering: Extract date features for prediction (ensure 'Date' is in datetime format)
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Simple linear regression model for temperature prediction
X = df[['Year', 'Month']]
y = df['Temp_2m']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict future temperature
future_years = np.array([2026, 2027, 2028]).reshape(-1, 1)
future_months = np.array([1, 6, 12]).reshape(-1, 1)  # Example months for prediction

future_dates = np.hstack([future_years, future_months])
predictions = model.predict(future_dates)

st.title("Temperature Prediction")
st.write("This page predicts future temperature trends.")

# Display predictions for the future
st.subheader("Predicted Future Temperatures")
for year, month, pred in zip(future_years.flatten(), future_months.flatten(), predictions):
    st.write(f"Predicted Temperature for {month}/{year}: {pred:.2f}°C")

# Plot the predictions
plt.plot(df['Date'], df['Temp_2m'], label="Historical Temperature")
plt.plot(future_dates[:, 0], predictions, label="Predicted Temperature", color='red', linestyle='--')
plt.title("Temperature Prediction")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
st.pyplot(plt)
