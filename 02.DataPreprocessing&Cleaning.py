# Practical 2: Data Preprocessing and Cleaning

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

df = pd.read_csv("datasets/data.csv")

print("Original Dataset:")
print(df.head())

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Fill missing values with mean
df["Calories"].fillna(df["Calories"].mean(), inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


print("\nDuplicate Rows Before Removal:", df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicate Rows After Removal:", df.duplicated().sum())


# Normalization (Min-Max Scaling)
minmax_scaler = MinMaxScaler()

df_normalized = df.copy()
df_normalized[["Duration", "Pulse", "Maxpulse", "Calories"]] = (
    minmax_scaler.fit_transform(
        df[["Duration", "Pulse", "Maxpulse", "Calories"]]
    )
)

print("\nNormalized Data:")
print(df_normalized.head())

# Standardization (Z-Score Scaling)
standard_scaler = StandardScaler()

df_standardized = df.copy()
df_standardized[["Duration", "Pulse", "Maxpulse", "Calories"]] = (
    standard_scaler.fit_transform(
        df[["Duration", "Pulse", "Maxpulse", "Calories"]]
    )
)

print("\nStandardized Data:")
print(df_standardized.head())


sample_df = pd.DataFrame({
    "Gender": ["Male", "Female", "Male", "Female"]
})

encoder = LabelEncoder()

sample_df["Gender_Encoded"] = encoder.fit_transform(sample_df["Gender"])

print("\nCategorical Variable Encoding:")
print(sample_df)


Q1 = df["Calories"].quantile(0.25)
Q3 = df["Calories"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df["Calories"] < lower_bound) |
    (df["Calories"] > upper_bound)
]

print("\nOutliers Detected:")
print(outliers)

print("\nNumber of Outliers:", len(outliers))

print("\nPreprocessing Completed Successfully!")