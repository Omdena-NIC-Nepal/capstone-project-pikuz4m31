import pandas as pd

# Loading dataset
df = pd.read_csv("../../data/raw/dailyclimate.csv")

# Dropping Unnecessary Column
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Drop duplicates
df.drop_duplicates(inplace=True)


# Outlier detection using IQR method
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] < lower) | (data[column] > upper)]

# Outlier detection to all numeric columns
numeric_cols = df.select_dtypes(include='number').columns
outliers = {}

for col in numeric_cols:
    outliers_df = detect_outliers_iqr(df, col)
    if not outliers_df.empty:
        outliers[col] = len(outliers_df)


df.to_csv("../../data/processed/dailyclimate_cleaned.csv", index=False)
print("Cleaned Data Successfully Saved")