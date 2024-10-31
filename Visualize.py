import time
import warnings

import altair as alt
import numpy as np
import streamlit as st
import pandas as pd

from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
from diana import DianaClustering

warnings.filterwarnings("ignore")


def generate_data(dataset_name, n_samples=50, seed=30):    
    rng = np.random.RandomState(seed)  
    
    if dataset_name == "noisy_circles":
        data = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)
        title = "Noisy Circles"
    elif dataset_name == "noisy_moons":
        data = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
        title = "Noisy Moons"
    elif dataset_name == "blobs":
        data = datasets.make_blobs(n_samples=n_samples, random_state=seed)
        title = "Blobs"
    elif dataset_name == "no_structure":
        data = (rng.rand(n_samples, 2), None)
        title = "No Structure"
    elif dataset_name == "aniso":
        X, y = datasets.make_blobs(n_samples=n_samples, random_state=seed)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_aniso = np.dot(X, transformation)
        data = (X_aniso, y)
        title = "Anisotropic Blobs"
    elif dataset_name == "varied":
        data = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=seed)
        title = "Blobs with Varied Variances"
    else:
        raise ValueError("Unknown dataset name.")
    return np.array(data[0]), np.array(data[1]), title


def apply_clustering(model_name, data, n_clusters, metric="euclidean", linkage="ward"):
    if model_name == "AGNES":
        model = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters, metric=metric)
    elif model_name == "DIANA":
        # model = AgglomerativeClustering(linkage="complete", n_clusters=n_clusters)
        model = DianaClustering(data, metric)
        cluster_labels = model.fit(n_clusters)
        return cluster_labels, model.children_
    elif model_name == "BIRCH":
        model = Birch(n_clusters=n_clusters)
        return model.fit_predict(data), None
    elif model_name == "Probabilistic":
        model = GaussianMixture(n_components=n_clusters)  # Number of clusters set to 2
        model.fit(data)
        return model.predict(data), None
    else:
        raise ValueError("Model not implemented or available.")
    
    return model.fit_predict(data), model.children_

def find_cluster(element, clusters):
    for idx, cluster in enumerate(clusters):
        if element in cluster:
            return idx
    return -1

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

def main():
    st.title("Visualization clustering")
    container = st.container()
    container.write("### Generated Data")
    col1, col2 = container.columns(2)

    data_types = {
        "Noisy Circles": "noisy_circles",
        "Noisy Moons": "noisy_moons",
        "Blobs": "blobs",
        "No Structure": "no_structure",
        "Anisotropic Blobs": "aniso",
        "Blobs with Varied Variances": "varied"
    }

    with col1:
        option = st.selectbox("Choose data type?",list(data_types.keys()),)
        
            
    with col2:
        numberDatapoints = st.slider(
        "Number of datapoint",
        min_value=10,
        ) 
        
    if container.button("Run", type = 'primary'): 
        data, labels, title = generate_data(data_types[option], numberDatapoints)
        data = pd.DataFrame(data, columns=["x","y"]) 

        st.session_state['data'] = data  # Store generated data in session state
        st.session_state['labels'] = labels  # Store generated labels in session state
        st.session_state['title'] = title  # Store title in session state

        
    
    # Check if data is generated and stored in session state
    if 'data' in st.session_state:
        data = st.session_state['data']
        title = st.session_state['title']
        labels = st.session_state['labels']
        container.scatter_chart(data, x="x", y="y")
        container.write("### Select a Clustering Model and Run")
        # Model selection dropdown

        container1 = st.container()
        container2 = st.container()
    
        sub_col1, sub_col2 = container1.columns(2)
        
        with sub_col1:
            model_option = st.selectbox("Choose clustering model", ("AGNES", "DIANA", "BIRCH", "Probabilistic"))
            
        with sub_col2:
            n_clusters = st.number_input(
                            "Number of clusters",
                            min_value=2,
                            ) 
            
        metric_key = {
                "euclidean": "euclidean",
                "manhattan":"manhattan",
                "cosine":"cosine"                
            }
        metric_type="euclidean"
        
            
        linkage_key = {"ward": "ward", "complete": "complete", "average": "average", "single": "single"}  
        linkage =  "complete" 
            
        if model_option == "AGNES" :
            s1, s2 = container2.columns(2)
            with s1:
                metric_type = st.selectbox("Choose metric type?",list(metric_key.keys()))                 
            with s2:
                linkage = st.selectbox("Choose linkage type?",list(linkage_key.keys())) 
            
        elif  model_option == "DIANA":
            metric_type = container2.selectbox("Choose metric type?",list(metric_key.keys())) 
        
        # Button to run clustering
        if st.button("Run Clustering"):
            # Apply clustering
            cluster_labels, model_data = apply_clustering(model_option, data, n_clusters, metric_type, linkage)
            data_pd = pd.DataFrame(data, columns=["x", "y"])
            chart_area = st.empty()
                    
            if model_data is not None and model_option=="AGNES":
                #Loop through the animation for agglo algorithm                
                labels = [i for i in range(numberDatapoints)]
                
                clusters = [[i] for i in range(numberDatapoints)]
                for i, (left, right) in enumerate(model_data):
                    new_cluster = clusters[left] + clusters[right]
                    relabel = min(labels[clusters[left][0]], labels[clusters[right][0]])
                    for ele in new_cluster:
                        labels[ele] = relabel
                    clusters.append(new_cluster)
                    clusters[left] = []
                    clusters[right] = []

                    data_pd["cluster"] = labels
                    chart = alt.Chart(data_pd).mark_circle(size=200).encode(
                        x="x",
                        y="y",
                        color="cluster:N",
                        tooltip=["x", "y", "cluster"]
                        ).properties(
                            width=700,
                            height=500,
                            title=f"{title} - Clustering with {model_option}"
                    ).interactive()
                    
                    chart_area.altair_chart(chart, use_container_width=True)
                    if i > numberDatapoints - n_clusters - 2:
                        break
                    time.sleep(0.7)
            elif model_data is not None and model_option=="DIANA":
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
                    
                    chart_area.altair_chart(chart, use_container_width=True)
                    time.sleep(1.5)

                    if len(np.unique(labels)) > n_clusters:
                        break
            else:
                data_pd["cluster"] = cluster_labels
                chart = alt.Chart(data_pd).mark_circle(size=200).encode(
                        x="x",
                        y="y",
                        color="cluster:N",
                        tooltip=["x", "y", "cluster"]
                        ).properties(
                            width=700,
                            height=500,
                            title=f"{title} - Clustering with {model_option}"
                    ).interactive()
                chart_area.altair_chart(chart, use_container_width=True)                 


if __name__ == "__main__":
    main()  
