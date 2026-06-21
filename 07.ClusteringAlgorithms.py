# Practical 7: Clustering Algorithms

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage,dendrogram,fcluster

df = pd.read_csv("datasets/data.csv")

df["Calories"] = df["Calories"].fillna(df["Calories"].mean())

X = df[["Duration", "Pulse", "Maxpulse", "Calories"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("===== K-MEANS CLUSTERING =====")

kmeans = KMeans(n_clusters=3,random_state=42,n_init=10)

clusters = kmeans.fit_predict(X_scaled)

df["KMeans_Cluster"] = clusters

sil_score = silhouette_score(X_scaled,clusters)

print("Silhouette Score:", sil_score)


plt.figure(figsize=(8,6))

plt.scatter(df["Duration"],df["Calories"],c=clusters)

plt.xlabel("Duration")
plt.ylabel("Calories")
plt.title("K-Means Clustering")

plt.show()

print("\n===== HIERARCHICAL CLUSTERING =====")


linkage_matrix = linkage(X_scaled,method="ward")

plt.figure(figsize=(10,6))

dendrogram(linkage_matrix)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")

plt.show()


hier_clusters = fcluster(linkage_matrix,3,criterion="maxclust")
df["Hierarchical_Cluster"] = hier_clusters

hier_score = silhouette_score(X_scaled,hier_clusters)
print("Silhouette Score:", hier_score)


print("\nK-Means Cluster Counts:")
print(df["KMeans_Cluster"].value_counts())

print("\nHierarchical Cluster Counts:")
print(df["Hierarchical_Cluster"].value_counts())

print("\nClustering Completed Successfully!")