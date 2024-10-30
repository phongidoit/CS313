# Import necessary libraries
import numpy as np
import pandas as pd
import altair as alt
import time
import streamlit as st

from diana import DianaClustering


# Function to apply clustering
def apply_clustering(model_class, data, n_clusters):
    model = model_class(data)
    cluster_labels = model.fit(n_clusters)
    return cluster_labels, model.children_

# Streamlit UI Code
st.title("Top-Down DIANA Clustering Animation")
st.sidebar.header("Clustering Parameters")
n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)

# Sample 2D data for clustering
data = np.random.rand(20, 2)  # Replace with your own data
data_pd = pd.DataFrame(data, columns=["x", "y"])
chart_area = st.empty()

# Initialize clustering
cluster_labels, model_data = apply_clustering(DianaClustering, data, n_clusters)

# Start with all points in a single cluster
labels = np.zeros(len(data), dtype=int)
data_pd["cluster"] = labels

# Top-Down Animation of DIANA Clustering
for i, (splinters, remaining) in enumerate(model_data):
    # Assign new labels for the split
    new_cluster_label = max(labels) + 1
    for idx in splinters:
        labels[idx] = new_cluster_label  # Assign new cluster to splinters
    
    data_pd["cluster"] = labels
    chart = alt.Chart(data_pd).mark_circle(size=200).encode(
        x="x",
        y="y",
        color="cluster:N",
        tooltip=["x", "y", "cluster"]
    ).properties(
        width=700,
        height=500,
        title=f"DIANA Clustering - Step {i+1}"
    ).interactive()
    
    chart_area.altair_chart(chart)
    time.sleep(2)

    if len(np.unique(labels)) >= n_clusters:
        break
