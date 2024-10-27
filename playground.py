import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def generate_data():
   return np.random.rand(7,2)

def find_cluster(element, clusters):
    for idx, cluster in enumerate(clusters):
        if element in cluster:
            return idx
    return -1
   
    
iris = load_iris()
X = generate_data()
X_pd = pd.DataFrame(X, columns=["x","y"])
# print(X.shape)
X_pd['color'] = X_pd.index
# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(n_clusters=None, distance_threshold=0)

model = model.fit(X)
result = model.fit_predict(X)


n_samples = len(X)
clusters = [[i] for i in range(n_samples)]
for i, (left, right) in enumerate(model.children_):
    new_cluster = clusters[left] + clusters[right]
    clusters.append(new_cluster)
    clusters[left] = []
    clusters[right] = []
    
    # Filter out empty clusters and print the current state
    active_clusters = [c for c in clusters if c]
    print(f"Iteration {i+1}: Clusters - {active_clusters}")
    for index in range (n_samples):
            X_pd.at[index, "color"] = find_cluster(index, active_clusters)
            print(X_pd.color)
            
    plt.title("Scatter plot " + str(i))
    sns.scatterplot(data=X_pd, x='x', y='y', hue='color', palette = "Paired")
    plt.show()        
            
            
    if len(active_clusters) <= 3:
        break
    
    
# plt.title("Final plot")   
# sns.scatterplot(data=X_pd, x='x', y='y', hue='color')
#print(result)

plt.show()

    
# plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
# plot_dendrogram(model, truncate_mode="level", p=3)
# plt.xlabel("Number of points in node (or index of point if no parenthesis).")
# plt.show()
