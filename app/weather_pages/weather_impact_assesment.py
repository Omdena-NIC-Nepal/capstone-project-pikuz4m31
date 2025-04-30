# import os
# import pandas as pd
# import sys

# # Print the current working directory to debug paths
# print("Current Working Directory:", os.getcwd())

# # Construct the correct path to the feature-engineered data file directly here
# file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'feature_engineering', 'weather_and_temp_feature_engineering.csv')

# # Print constructed path and verify it exists
# print("Constructed file path:", file_path)

# # Check if the file exists
# if not os.path.exists(file_path):
#     print(f"❌ File not found at: {file_path}")
#     sys.exit(1)
# else:
#     print(f"✅ File found at: {file_path}")

# # Load the data directly in weather impact assessment
# df = pd.read_csv(file_path)
# print("Data loaded successfully!")

# # Proceed with any additional operations on the `df` here
# # For example, data inspection or processing specific to weather impact
# # print(df.head())  # Example to inspect the first few rows of data

# # Now, you can continue with your logic in the weather impact assessment script.
# # Do NOT reference the modeling functions here. Just process the data as needed.



# import os
# import pandas as pd
# import sys

# # Print the current working directory to debug paths
# print("Current Working Directory:", os.getcwd())

# # Construct the correct path to the feature-engineered data file directly here
# file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'feature_engineering', 'weather_and_temp_feature_engineering.csv')

# # Print constructed path and verify it exists
# print("Constructed file path:", file_path)

# # Check if the file exists
# if not os.path.exists(file_path):
#     print(f"❌ File not found at: {file_path}")
#     sys.exit(1)
# else:
#     print(f"✅ File found at: {file_path}")

# # Load the data directly in weather impact assessment
# df = pd.read_csv(file_path)
# print("Data loaded successfully!")

# # Extract relevant columns for weather impact analysis
# weather_impact_features = [
#     'latitude', 'longitude', 'temperature_avg', 'precipitation',
#     'precip_rolling_30d_mean', 'precip_rolling_30d_std', 'spi_like',
#     'heat_stress_index', 'is_monsoon', 
#     'temp_lag_1', 'precip_lag_1',
#     'temp_lag_3', 'precip_lag_3',
#     'temp_lag_7', 'precip_lag_7',
#     'temp_lag_30', 'precip_lag_30',
#     'distance_to_center_km', 
#     'pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5',
#     'pca_6', 'pca_7', 'pca_8', 'pca_9', 'pca_10'
# ]

# # Select only the relevant columns for the weather impact assessment
# df_weather_impact = df[weather_impact_features]

# # Example: Inspect the first few rows of the selected data
# print("First few rows of the selected weather impact features:")
# print(df_weather_impact.head())

# # Proceed with any additional operations on the `df_weather_impact` here
# # This could involve further processing, data visualization, or other analyses as needed

# # You can also call feature engineering CSV here if you need to reprocess or reapply some transformations
# # For example, you might need to clean, impute missing values, or rescale features based on new requirements


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# from geopy.distance import geodesic

# # Load the dataset
# df = pd.read_csv('./../../feature_engineering/weather_and_temp_feature_engineering.csv')

# # Ensure 'date' column is datetime
# df['date'] = pd.to_datetime(df['date'])

# # 1. **Precipitation Patterns (Rolling Averages and Deviations)**

# st.write("### Precipitation Patterns (Rolling Averages and Deviations)")
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(df['date'], df['precip_rolling_30d_mean'], label='30-Day Rolling Average', color='tab:blue')
# ax.plot(df['date'], df['precip_rolling_30d_std'], label='30-Day Rolling Std Dev', color='tab:red', linestyle='--')
# ax.set_xlabel('Date')
# ax.set_ylabel('Precipitation (mm)')
# ax.set_title('Precipitation Rolling Averages and Deviations')
# ax.legend(loc='upper left')
# st.pyplot(fig)

# # 2. **Temperature Fluctuations (Including Heat Stress and Lag Features)**

# st.write("### Temperature Fluctuations (Including Heat Stress and Lag Features)")

# # Temperature fluctuation time series
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(df['date'], df['temperature_avg'], label='Temperature Avg', color='tab:red')
# ax.set_xlabel('Date')
# ax.set_ylabel('Temperature (°C)')
# ax.set_title('Temperature Fluctuations Over Time')
# ax.legend(loc='upper left')
# st.pyplot(fig)

# # Heat Stress (if humidity data exists)
# if 'humidity' in df.columns:
#     df['heat_stress_index'] = df['humidity'] * df['temperature_avg']
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(df['date'], df['heat_stress_index'], label='Heat Stress Index', color='tab:orange')
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Heat Stress Index')
#     ax.set_title('Heat Stress Index Over Time')
#     ax.legend(loc='upper left')
#     st.pyplot(fig)

# # Lag Features for Temperature and Precipitation
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(df['date'], df['temp_lag_1'], label='Temperature Lag 1 Day', color='tab:purple')
# ax.plot(df['date'], df['precip_lag_1'], label='Precipitation Lag 1 Day', color='tab:green')
# ax.set_xlabel('Date')
# ax.set_ylabel('Lag Features')
# ax.set_title('Lag Features for Temperature and Precipitation')
# ax.legend(loc='upper left')
# st.pyplot(fig)

# # 3. **Seasonal Patterns (Monsoon Indicators)**

# st.write("### Seasonal Patterns (Monsoon Indicators)")

# # Monsoon indicator visualization
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(df['date'], df['is_monsoon'], label='Monsoon Indicator (1=Monsoon)', color='tab:blue')
# ax.set_xlabel('Date')
# ax.set_ylabel('Monsoon (1 = Monsoon, 0 = Not Monsoon)')
# ax.set_title('Monsoon Indicators Over Time')
# ax.legend(loc='upper left')
# st.pyplot(fig)

# # 4. **Spatial Considerations (Distance from the Center)**

# st.write("### Spatial Considerations (Distance from the Center)")

# # Calculating the distance from the center (mean latitude and longitude)
# ref_point = (df['latitude'].mean(), df['longitude'].mean())
# df['distance_to_center_km'] = df.apply(lambda row: geodesic((row['latitude'], row['longitude']), ref_point).km, axis=1)

# # Visualize the spatial distribution of distances from the center
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.scatter(df['latitude'], df['longitude'], c=df['distance_to_center_km'], cmap='viridis', s=20)
# ax.set_xlabel('Latitude')
# ax.set_ylabel('Longitude')
# ax.set_title('Spatial Distribution of Distance from Center')
# st.pyplot(fig)

# # 5. **Principal Components (Capturing the Most Important Climate Variations)**

# st.write("### Principal Components (Capturing the Most Important Climate Variations)")

# # PCA Components 1, 2, 3 Visualization
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(df['pca_1'], df['pca_2'], df['pca_3'], c=df['latitude'], cmap='viridis', s=20)
# ax.set_xlabel('PCA 1')
# ax.set_ylabel('PCA 2')
# ax.set_zlabel('PCA 3')
# ax.set_title('PCA Components 1, 2, 3')
# st.pyplot(fig)


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from geopy.distance import geodesic
from mpl_toolkits.mplot3d import Axes3D
import os
# Load and clean dataset
# df = pd.read_csv('../feature_engineering/weather_and_temp_feature_engineering.csv')
# Get current file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '../feature_engineering/weather_and_temp_feature_engineering.csv'))

# Load data
df = pd.read_csv(DATA_PATH)
# Load dataset
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')

st.title("🌦️ Weather Impact Assessment Dashboard")

# Sidebar Filters
st.sidebar.header("Filter Options")
districts = df['district'].dropna().unique().tolist()
selected_district = st.sidebar.selectbox("Select a District", ['All'] + districts)
if selected_district != 'All':
    df = df[df['district'] == selected_district]

# 1. Precipitation Trends (Rolling Averages)
st.subheader("📉 Precipitation Rolling Patterns")
fig1, ax1 = plt.subplots()
ax1.plot(df['date'], df['precip_rolling_30d_mean'], label='30-Day Avg', color='blue')
ax1.plot(df['date'], df['precip_rolling_30d_std'], label='30-Day Std Dev', color='red', linestyle='--')
ax1.set(title='Rolling Precipitation Metrics', xlabel='Date', ylabel='Scaled Precipitation')
ax1.legend()
st.pyplot(fig1)

# 2. Drought & Heat Stress (SPI and HSI)
st.subheader("🔥 Drought & Heat Stress Trends")
fig2, ax2 = plt.subplots()
ax2.plot(df['date'], df['spi_like'], label='SPI-like (Drought Index)', color='brown')
ax2.plot(df['date'], df['heat_stress_index'], label='Heat Stress Index', color='orange')
ax2.legend()
ax2.set(title='SPI & Heat Stress Trends', xlabel='Date', ylabel='Index')
st.pyplot(fig2)

# 3. Temperature & Lag Effects
st.subheader("🌡️ Temperature & Lag Effects")
fig3, ax3 = plt.subplots()
ax3.plot(df['date'], df['temperature_avg'], label='Avg Temp', color='tomato')
ax3.plot(df['date'], df['temp_lag_1'], label='Temp Lag 1', color='purple', linestyle=':')
ax3.plot(df['date'], df['temp_lag_30'], label='Temp Lag 30', color='gray', linestyle='--')
ax3.set(title='Temperature & Lag Patterns', xlabel='Date')
ax3.legend()
st.pyplot(fig3)

# 4. Monsoon Impact
st.subheader("🌧️ Monsoon Impact")
fig4 = px.line(df, x='date', y='is_monsoon', color='district' if selected_district == 'All' else None,
               title='Monsoon Seasonality')
st.plotly_chart(fig4)

# 5. Seasonal Disaster Pattern
st.subheader("📆 Seasonal Disaster Occurrence")
if 'disaster_type' in df.columns and df['disaster_type'].notna().any():
    seasonal_df = df[df['disaster_type'].notna()]
    seasonal_df['month'] = seasonal_df['date'].dt.month
    fig5 = px.histogram(seasonal_df, x='month', color='disaster_type', barmode='group',
                        title='Disaster Frequency by Month')
    st.plotly_chart(fig5)
else:
    st.info("No disaster occurrence data available for seasonal analysis.")

# 6. Geospatial Vulnerability
st.subheader("🗺️ Spatial Risk Visualization")
ref_point = (df['latitude'].mean(), df['longitude'].mean())
df['distance_to_center_km'] = df.apply(lambda row: geodesic((row['latitude'], row['longitude']), ref_point).km, axis=1)
fig6 = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='distance_to_center_km',
                         hover_data=['date'], zoom=4, mapbox_style='carto-positron',
                         title="Distance from Center (Geospatial Spread)")
st.plotly_chart(fig6)

# 7. Principal Component Impact
st.subheader("📊 PCA Components (Regional Variation)")
fig7 = px.scatter_3d(df, x='pca_1', y='pca_2', z='pca_3', color='latitude',
                     title="PCA Component Space", opacity=0.6)
st.plotly_chart(fig7)

# 8. Cluster-like Behavior via Heatmap
st.subheader("🧩 Clustering Insight via Correlation")
corr_cols = ['temperature_avg', 'precipitation', 'spi_like', 'heat_stress_index', 'precip_rolling_30d_mean']
fig8, ax8 = plt.subplots()
sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', ax=ax8)
ax8.set_title("Feature Correlation Heatmap")
st.pyplot(fig8)

# 9. Disaster-Lag Correlation
st.subheader("📈 Lag Features & Disaster Co-occurrence")
if 'disaster_type' in df.columns:
    disaster_df = df[df['disaster_type'].notna()]
    fig9 = px.scatter(disaster_df, x='temp_lag_7', y='precip_lag_7', color='disaster_type',
                      title='Lag Features at Time of Disasters')
    st.plotly_chart(fig9)

# 10. Full Table View (Toggle)
if st.checkbox("Show Full Dataset"):
    st.write(df)

