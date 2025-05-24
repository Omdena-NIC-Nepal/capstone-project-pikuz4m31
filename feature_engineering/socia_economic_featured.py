import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from shapely.geometry import Point

# =====================
# 1. Load and Clean Data
# =====================
socio_df = pd.read_csv("../data/processed/cleaned_soci_economic_profile.csv")
geo_df = gpd.read_file("../data/processed/geo_socio_economic_profile.geojson")

# Rename and subset relevant columns
socio_df = socio_df[[
    'district_matched_second_round', 'Year',
    'Average Annual Rainfall(in mm)', 'Altitude in masl',
    'Estimated Loss (Million NRs.)', 'Number of Affected Family'
]].rename(columns={
    'district_matched_second_round': 'district',
    'Year': 'year',
    'Average Annual Rainfall(in mm)': 'rainfall_mm',
    'Altitude in masl': 'altitude_m',
    'Estimated Loss (Million NRs.)': 'loss_million_nrs',
    'Number of Affected Family': 'affected_families'
})

# Parse year properly
socio_df['year'] = pd.to_datetime(socio_df['year'], errors='coerce').dt.year
socio_df = socio_df.dropna(subset=['year'])
socio_df['year'] = socio_df['year'].astype(int)

# Convert to numeric
socio_df['rainfall_mm'] = pd.to_numeric(socio_df['rainfall_mm'], errors='coerce')
socio_df['altitude_m'] = pd.to_numeric(socio_df['altitude_m'], errors='coerce')

# =====================
# 2. Derived Climate Indices
# =====================
socio_df['drought_index'] = socio_df['altitude_m'] / (socio_df['rainfall_mm'] + 1)
socio_df['heat_stress_index'] = socio_df['rainfall_mm'] / (socio_df['altitude_m'] + 1)

# =====================
# 3. Seasonal (Monsoon) Indicators
# =====================
socio_df['monsoon_season'] = socio_df['year'].apply(lambda y: 1 if y % 4 == 0 else 0)

# =====================
# 4. Lag Features (by district)
# =====================
socio_df = socio_df.sort_values(by=['district', 'year'])
socio_df['rainfall_lag1'] = socio_df.groupby('district')['rainfall_mm'].shift(1)
socio_df['drought_index_lag1'] = socio_df.groupby('district')['drought_index'].shift(1)

# =====================
# 5. Spatial Proximity Features (Fixing invalid geometries)
# =====================
# Drop invalid geometries
geo_df = geo_df[geo_df.is_valid & geo_df.geometry.notnull()].copy()
geo_df = geo_df.to_crs(epsg=32645)
geo_df['centroid'] = geo_df.geometry.centroid
geo_df['longitude'] = geo_df.centroid.x
geo_df['latitude'] = geo_df.centroid.y

# Merge with socio_df
merged = pd.merge(socio_df, geo_df[['district_matched', 'longitude', 'latitude']],
                  left_on='district', right_on='district_matched', how='left')

# Reference point: Kathmandu in UTM zone 45N (approx.)
kathmandu = Point(500000, 3060000)
merged['distance_to_kathmandu'] = merged.apply(
    lambda row: kathmandu.distance(Point(row['longitude'], row['latitude']))
    if pd.notnull(row['longitude']) and pd.notnull(row['latitude']) else np.nan, axis=1
)

# =====================
# 6. Satellite Features (placeholder)
# =====================
merged['ndvi_placeholder'] = np.random.uniform(0.2, 0.8, size=len(merged))

# =====================
# 7. Normalize and Scale with Imputation (fixing column NaNs)
# =====================
numeric_cols = [
    'rainfall_mm', 'altitude_m', 'loss_million_nrs', 'affected_families',
    'drought_index', 'heat_stress_index', 'rainfall_lag1',
    'drought_index_lag1', 'distance_to_kathmandu', 'ndvi_placeholder'
]

# Drop columns that are entirely NaN
valid_numeric_cols = [col for col in numeric_cols if merged[col].notnull().sum() > 0]

# Impute missing values
imputer = SimpleImputer(strategy='mean')
merged[valid_numeric_cols] = imputer.fit_transform(merged[valid_numeric_cols])

# Scale
scaler = StandardScaler()
merged_scaled = merged.copy()
merged_scaled[valid_numeric_cols] = scaler.fit_transform(merged[valid_numeric_cols])

# =====================
# 8. Dimensionality Reduction (only on valid columns)
# =====================
pca = PCA(n_components=3)
pca_features = pca.fit_transform(merged_scaled[valid_numeric_cols])
merged_scaled[['pca1', 'pca2', 'pca3']] = pca_features

# =====================
# 9. Save the Processed File
# =====================
merged_scaled.to_csv("feature_engineered_socio_climate.csv", index=False)
print("✅ Feature engineering complete. Output saved to 'feature_engineered_socio_climate.csv'")
