import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("../data/processed/dailyclimate_cleaned.csv")

# Display basic info
st.title("Climate Data Analysis")
st.write("This page allows you to analyze trends in the climate data.")

# Temperature and Precipitation Analysis
st.subheader("Temperature Analysis")
st.write(f"Mean Temperature: {df['Temp_2m'].mean():.2f}°C")
st.write(f"Temperature Range: {df['Temp_2m'].min():.2f}°C - {df['Temp_2m'].max():.2f}°C")

st.subheader("Precipitation Analysis")
st.write(f"Mean Precipitation: {df['Precip'].mean():.2f}mm")
st.write(f"Precipitation Range: {df['Precip'].min():.2f}mm - {df['Precip'].max():.2f}mm")

# Plot the distribution of temperature and precipitation
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.histplot(df['Temp_2m'], bins=30, kde=True, ax=ax[0], color="skyblue")
ax[0].set_title("Temperature Distribution")

sns.histplot(df['Precip'], bins=30, kde=True, ax=ax[1], color="green")
ax[1].set_title("Precipitation Distribution")

st.pyplot(fig)

# Yearly Temperature Trends
st.subheader("Yearly Temperature Trends")

# Ensure 'Date' column is datetime
if not pd.api.types.is_datetime64_any_dtype(df['Date']):
    df['Date'] = pd.to_datetime(df['Date'])

# Extract year and compute average temperature per year
df['Year'] = df['Date'].dt.year
yearly_avg_temp = df.groupby('Year')['Temp_2m'].mean().reset_index()

# Plot yearly trend
fig_yearly = plt.figure(figsize=(10, 5))
sns.lineplot(data=yearly_avg_temp, x='Year', y='Temp_2m', marker='o', color='tomato')
plt.title("Average Yearly Temperature")
plt.xlabel("Year")
plt.ylabel("Mean Temp (°C)")
plt.grid(True)

st.pyplot(fig_yearly)

# Outlier Detection
st.subheader("Outlier Detection (IQR Method)")
numeric_cols = df.select_dtypes(include='number').columns
outliers = {}

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers_df = df[(df[col] < lower) | (df[col] > upper)]
    if not outliers_df.empty:
        outliers[col] = len(outliers_df)

# st.write(f"Outliers detected in columns: {outliers}")


# Boxplots for Visualizing Outliers
st.subheader("Boxplot Visualization of Outliers")

fig_outliers, axs = plt.subplots(len(numeric_cols), 1, figsize=(10, 5 * len(numeric_cols)))

# Handle single subplot case
if len(numeric_cols) == 1:
    axs = [axs]

for i, col in enumerate(numeric_cols):
    sns.boxplot(x=df[col], ax=axs[i], color='lightcoral')
    axs[i].set_title(f"Boxplot for {col}")
    axs[i].set_xlabel(col)

st.pyplot(fig_outliers)