# Practical 8: Principal Component Analysis (PCA)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


df = pd.read_csv("datasets/data.csv")
df["Calories"] = df["Calories"].fillna(df["Calories"].mean())


X = df[["Duration", "Pulse", "Maxpulse", "Calories"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca_2d = PCA(n_components=2)

X_pca_2d = pca_2d.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca_2d,columns=["PC1", "PC2"])

print("Explained Variance Ratio (2D):")
print(pca_2d.explained_variance_ratio_)

print("\nTotal Variance Explained:")
print(sum(pca_2d.explained_variance_ratio_))


plt.figure(figsize=(8,6))

plt.scatter(pca_df["PC1"],pca_df["PC2"])

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - 2D Visualization")

plt.show()

pca_3d = PCA(n_components=3)

X_pca_3d = pca_3d.fit_transform(X_scaled)

print("\nExplained Variance Ratio (3D):")
print(pca_3d.explained_variance_ratio_)

print("\nTotal Variance Explained:")
print(sum(pca_3d.explained_variance_ratio_))


fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_pca_3d[:,0],X_pca_3d[:,1],X_pca_3d[:,2])

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plt.title("PCA - 3D Visualization")
plt.show()


print("\nPCA Components:")
print(pca_2d.components_)

print("\nDimensionality Reduction Completed Successfully!")