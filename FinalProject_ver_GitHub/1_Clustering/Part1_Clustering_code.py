import pandas as pd
from matplotlib import rcParams
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.manifold import TSNE
import umap

# Set font to SimSun and Times New Roman
config = {
    "font.family": 'serif',
    "font.size": 18,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

plt.rcParams['axes.unicode_minus'] = False

# Load the dataset
file_path = r"1_Clustering\data_risk_Industry.csv"
data = pd.read_csv(file_path)

# Filter data between 2009-01-09 and 2010-01-09 and use the 'Industry_Label' column
filtered_data = data[(data['Date'] >= '2009-01-09') & (data['Date'] <= '2010-01-09')]
filtered_data = filtered_data[['Date', 'Ind', 'Industry_Label', 'Risk']]

# Pivot data to have industries as columns and dates as rows
pivot_data = filtered_data.pivot(index='Date', columns='Industry_Label', values='Risk')
pivot_data = pivot_data.interpolate()  # Fill missing values

# ----------------Spectral Clustering----------------
n_clusters = 5  # Number of clusters
sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
cluster_labels_spectral = sc.fit_predict(pivot_data.T)  # Perform clustering on industries (columns)

# Add cluster results to the industry data
clustered_data_spectral = pd.DataFrame({'Ind': pivot_data.columns, 'Cluster': cluster_labels_spectral + 1})

# Plot clustering results with a horizontal axis interval of 1
plt.figure(figsize=(20, 12))
sns.scatterplot(data=clustered_data_spectral, x='Ind', y='Cluster', hue='Cluster', palette='viridis', s=100)
plt.title('Spectral Clustering Results of Industries (Numeric Labels)')
plt.xticks(ticks=range(len(clustered_data_spectral['Ind'])), labels=clustered_data_spectral['Ind'], rotation=45, fontsize=15)
plt.savefig("1_Clustering/graph_spectral_clustering.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# ----------------Iris Dendrogram (Hierarchical Clustering)----------------
# Perform hierarchical clustering to generate an Iris dendrogram
linked = linkage(pivot_data.T, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linked, labels=pivot_data.columns, orientation='top')
plt.title('Iris Dendrogram (Hierarchical Clustering of Industries)')
plt.xticks(rotation=0)
plt.savefig("1_Clustering/graph_iris_clustering.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Convert hierarchical clustering into clusters
hierarchical_clusters = fcluster(linked, t=n_clusters, criterion='maxclust')

# ----------------UMAP Dimensionality Reduction----------------
umap_reducer = umap.UMAP(n_neighbors=3, min_dist=0.3, random_state=42)
umap_embedding = umap_reducer.fit_transform(pivot_data.T)

# Plot UMAP dimensionality reduction results
plt.figure(figsize=(7, 6))
plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=cluster_labels_spectral, cmap='viridis', s=100)
plt.title('UMAP Dimensionality Reduction Results')
for i, label in enumerate(pivot_data.columns):
    plt.text(umap_embedding[i, 0], umap_embedding[i, 1], label, fontsize=9)
plt.savefig("1_Clustering/graph_umap.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# ----------------t-SNE Dimensionality Reduction----------------
tsne_reducer = TSNE(n_components=2, perplexity=8, random_state=20)
tsne_embedding = tsne_reducer.fit_transform(pivot_data.T)

# Plot t-SNE dimensionality reduction results
plt.figure(figsize=(7, 6))
plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=cluster_labels_spectral, cmap='viridis', s=100)
plt.title('t-SNE Dimensionality Reduction Results')
for i, label in enumerate(pivot_data.columns):
    plt.text(tsne_embedding[i, 0], tsne_embedding[i, 1], label, fontsize=14)
plt.savefig("1_Clustering/graph_tsne.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()
