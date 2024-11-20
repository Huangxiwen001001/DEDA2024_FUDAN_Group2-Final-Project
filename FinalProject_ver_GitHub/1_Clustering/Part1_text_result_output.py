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

# Path to save clustering results
result_file_path = r"1_Clustering\result_clustering.txt"

# Open the result file to write the clustering results
with open(result_file_path, 'w') as result_file:

    # ----------------Spectral Clustering----------------
    n_clusters = 5  # Number of clusters
    sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    cluster_labels_spectral = sc.fit_predict(pivot_data.T)  # Perform clustering on industries (columns)

    # Add cluster results to the industry data
    clustered_data_spectral = pd.DataFrame({'Industry_Label': pivot_data.columns, 'Cluster': cluster_labels_spectral + 1})
    # Merge with industry English names
    clustered_data_spectral = clustered_data_spectral.merge(filtered_data[['Industry_Label', 'Ind']].drop_duplicates(), on='Industry_Label')

    # Save the spectral clustering results to the file
    result_file.write("Spectral Clustering Results:\n")
    for cluster_num in range(1, n_clusters + 1):
        result_file.write(f"Cluster {cluster_num}:\n")
        cluster_data = clustered_data_spectral[clustered_data_spectral['Cluster'] == cluster_num]
        for _, row in cluster_data.iterrows():
            result_file.write(f'"{row["Industry_Label"]}: {row["Ind"]}"; ')
        result_file.write("\n")
    result_file.write("\n")

    # ----------------Iris Dendrogram (Hierarchical Clustering)----------------
    # Perform hierarchical clustering to generate an Iris dendrogram
    linked = linkage(pivot_data.T, method='ward')

    # Convert hierarchical clustering into clusters
    hierarchical_clusters = fcluster(linked, t=n_clusters, criterion='maxclust')

    # Add cluster results to the industry data
    clustered_data_hierarchical = pd.DataFrame({'Industry_Label': pivot_data.columns, 'Cluster': hierarchical_clusters})
    clustered_data_hierarchical = clustered_data_hierarchical.merge(filtered_data[['Industry_Label', 'Ind']].drop_duplicates(), on='Industry_Label')

    # Save the hierarchical clustering results to the file
    result_file.write("Hierarchical Clustering Results:\n")
    for cluster_num in range(1, n_clusters + 1):
        result_file.write(f"Cluster {cluster_num}:\n")
        cluster_data = clustered_data_hierarchical[clustered_data_hierarchical['Cluster'] == cluster_num]
        for _, row in cluster_data.iterrows():
            result_file.write(f'"{row["Industry_Label"]}: {row["Ind"]}"; ')
        result_file.write("\n")
    result_file.write("\n")

    # ----------------UMAP Dimensionality Reduction----------------
    umap_reducer = umap.UMAP(n_neighbors=3, min_dist=0.3, random_state=42)
    umap_embedding = umap_reducer.fit_transform(pivot_data.T)

    # Save UMAP dimensionality reduction results to the file
    result_file.write("UMAP Dimensionality Reduction Results:\n")
    for i, label in enumerate(pivot_data.columns):
        industry_name = filtered_data.loc[filtered_data['Industry_Label'] == label, 'Ind'].values[0]
        result_file.write(f'"{label}: {industry_name}"; Position: {umap_embedding[i]}\n')
    result_file.write("\n")

    # ----------------t-SNE Dimensionality Reduction----------------
    tsne_reducer = TSNE(n_components=2, perplexity=8, random_state=20)
    tsne_embedding = tsne_reducer.fit_transform(pivot_data.T)

    # Save t-SNE dimensionality reduction results to the file
    result_file.write("t-SNE Dimensionality Reduction Results:\n")
    for i, label in enumerate(pivot_data.columns):
        industry_name = filtered_data.loc[filtered_data['Industry_Label'] == label, 'Ind'].values[0]
        result_file.write(f'"{label}: {industry_name}"; Position: {tsne_embedding[i]}\n')
    result_file.write("\n")
