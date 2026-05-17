
# PROJET MACHINE LEARNING - SPOTIFY SONGS  


import os
os.environ["LOKY_MAX_CPU_COUNT"] = "2"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage


sns.set_style("whitegrid")


df = pd.read_csv("spotify.csv")
X = df.select_dtypes(include=np.number).dropna()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_score = silhouette_score(X_scaled, kmeans_labels)
print(f"Silhouette KMeans : {kmeans_score}")

dbscan = DBSCAN(eps=0.8, min_samples=15) 
dbscan_labels = dbscan.fit_predict(X_scaled)

if len(np.unique(dbscan_labels)) > 1:
    dbscan_score = silhouette_score(X_scaled, dbscan_labels)
else:
    dbscan_score = 0
print(f"Silhouette DBSCAN : {dbscan_score}")

X_sample = X_scaled[:5000] 
agg = AgglomerativeClustering(n_clusters=3)
agg_labels_sample = agg.fit_predict(X_sample)
agg_score = silhouette_score(X_sample, agg_labels_sample)
print(f"Silhouette Agglomerative (Sample) : {agg_score}")

algorithms = ["K-Means", "DBSCAN", "Agglomerative"]
scores = [kmeans_score, dbscan_score, agg_score]

plt.figure(figsize=(8,5))
sns.barplot(x=algorithms, y=scores, palette="viridis")
plt.ylabel("Silhouette Score")
plt.title("Comparaison des algorithmes")
plt.savefig("comparison.png", bbox_inches='tight')
plt.show()

linked = linkage(X_sample[:200], method='ward')
plt.figure(figsize=(12,6))
dendrogram(linked)
plt.title("Dendrogramme ")
plt.savefig("dendrogram.png", bbox_inches='tight')
plt.show()

tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(X_scaled[:5000]) # عينة لتسريع الرسم
plt.figure(figsize=(8,6))
plt.scatter(tsne_result[:,0], tsne_result[:,1], c=kmeans_labels[:5000], cmap='viridis')
plt.title("Visualisation t-SNE")
plt.savefig("tsne.png", bbox_inches='tight')
plt.show()


variance = np.var(X_scaled, axis=0)
importance = pd.DataFrame({"Feature": X.columns, "Importance": variance}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10,7))
sns.barplot(x="Importance", y="Feature", data=importance.head(10))
plt.title("Variables les plus importantes")
plt.savefig("feature_importance.png", bbox_inches='tight')
plt.show()


importances = np.var(X_scaled, axis=0)
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Importance des caractéristiques dans le clustering")
plt.bar(range(X.shape[1]), importances[indices], align="center", color='skyblue')
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.tight_layout()
plt.savefig("feature_importance1.png")
plt.show()

from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)

db_index = davies_bouldin_score(X_scaled, kmeans_labels)
ch_index = calinski_harabasz_score(X_scaled, kmeans_labels)

print(f"Davies-Bouldin Index: {db_index:.4f}")
print(f"Calinski-Harabasz Index: {ch_index:.4f}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='viridis', s=40)
plt.title("Clustering avec Gaussian Mixture Model")
plt.colorbar(label='Cluster')
plt.savefig("gmm_clustering.png")
plt.show()
plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=kmeans_labels,
    cmap='viridis',
    s=40
)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Projection PCA des données Spotify")
plt.colorbar(label='Cluster')
plt.savefig("pca_projection.png", bbox_inches='tight')
plt.show()
