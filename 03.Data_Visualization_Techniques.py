# Practical 3: Data Visualization Techniques

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("datasets/data.csv")

df["Calories"] = df["Calories"].fillna(df["Calories"].mean())


# ----------------------------------
# 2. Histogram
# ----------------------------------

plt.figure(figsize=(8,5))
plt.hist(df["Calories"], bins=10)
plt.title("Histogram of Calories")
plt.xlabel("Calories")
plt.ylabel("Frequency")
plt.show()

# ----------------------------------
# 3. Box Plot
# ----------------------------------

plt.figure(figsize=(8,5))
sns.boxplot(y=df["Calories"])
plt.title("Box Plot of Calories")
plt.show()

# ----------------------------------
# 4. Scatter Plot
# ----------------------------------

plt.figure(figsize=(8,5))
plt.scatter(df["Duration"], df["Calories"])
plt.title("Scatter Plot Between Duration and Calories")
plt.xlabel("Duration")
plt.ylabel("Calories")
plt.show()

# ----------------------------------
# 5. Correlation Matrix
# ----------------------------------

print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))

# ----------------------------------
# 6. Heatmap
# ----------------------------------

plt.figure(figsize=(6,4))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

# ----------------------------------
# 7. Pair Plot
# ----------------------------------

sns.pairplot(df)
plt.show()

print("Data Visualization Completed Successfully!")