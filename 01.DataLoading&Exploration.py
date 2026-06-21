# Practical 1: Data Loading and Exploration (EDA)

import pandas as pd

df = pd.read_csv("datasets/data.csv")

print("Dataset Loaded Successfully!\n")


print("First 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())


print("\nNumber of Rows and Columns:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)


print("\nDataset Information:")
df.info()

print("\nSummary Statistics:")
print(df.describe())


print("\nMissing Values in Each Column:")
print(df.isnull().sum())


print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

print("\n---- Missing Values (%)---")
print((df.isnull().sum()/len(df)) * 100)


print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())


print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))


print("\nUnique Values in Each Column:")
print(df.nunique())
