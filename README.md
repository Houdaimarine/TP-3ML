# Spotify Clustering Analysis using Machine Learning

Ce projet présente une étude comparative de plusieurs algorithmes de clustering appliqués à un dataset Spotify contenant différentes caractéristiques audio des chansons.

L’objectif principal est d’analyser automatiquement les similarités entre les morceaux musicaux afin de regrouper les chansons en clusters homogènes selon leurs propriétés audio.

Plusieurs techniques de Machine Learning non supervisé ont été utilisées, notamment :

- K-Means
- DBSCAN
- Agglomerative Clustering
- Gaussian Mixture Model (GMM)

Le projet inclut également une analyse statistique approfondie, des visualisations de données ainsi qu’une réduction de dimension avec PCA et t-SNE.

# Technologies utilisées

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

# Méthodologie

Les principales étapes du projet sont :

1. Chargement et nettoyage des données
2. Prétraitement et normalisation
3. Analyse statistique
4. Réduction de dimension avec PCA et t-SNE
5. Application des algorithmes de clustering
6. Évaluation des performances avec :
   - Silhouette Score
   - Davies-Bouldin Index
   - Calinski-Harabasz Index

# Résultats

Les résultats obtenus montrent que les performances des algorithmes varient selon la structure des données musicales.

Agglomerative Clustering a obtenu les meilleurs résultats selon le Silhouette Score, tandis que DBSCAN s’est montré efficace pour la détection des valeurs aberrantes.

#Visualisations réalisées

Le projet contient plusieurs visualisations :

- Histogrammes des variables
- Matrice de corrélation
- Boxplots
- Méthode du coude
- Projection PCA
- Visualisation t-SNE
- Dendrogramme
- Comparaison des performances

# Auteur

Houda Imarine  
Master Intelligence Artificielle
