import time
import warnings

import altair as alt
import numpy as np
import streamlit as st
import pandas as pd

from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture


warnings.filterwarnings("ignore")


def generate_data(dataset_name, n_samples=50, seed=30):
    rng = np.random.RandomState(seed)
    rng = np.random.RandomState(seed)
    
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


def apply_clustering(model_name, data, n_clusters):
    if model_name == "AGNES":
        model = AgglomerativeClustering(linkage="ward", n_clusters=n_clusters)
    elif model_name == "DIANA":
        model = AgglomerativeClustering(linkage="complete", n_clusters=n_clusters)
    elif model_name == "BIRCH":
        model = Birch(n_clusters=n_clusters)
    elif model_name == "Probabilistic":
        model = GaussianMixture(n_components=n_clusters)  # Number of clusters set to 2
        model.fit(data)
        return model.predict(data), model
    else:
        raise ValueError("Model not implemented or available.")
    
    return model.fit_predict(data), model

def find_cluster(element, clusters):
    for idx, cluster in enumerate(clusters):
        if element in cluster:
            return idx
    return -1

def get_label_each_iteration():
    pass

def main():
    st.title("Visualization clustering")
    container = st.container()
    
    col1, col2 = container.columns(2)

        
    with col1:
        option = st.selectbox("Choose data type?",("noisy_circles", "noisy_moons", "blobs", "no_structure", "aniso", ),)
        
            
    with col2:
        numberDatapoints = st.number_input(
        "Number of datapoint",
        min_value=50,
        ) 
        
    if container.button("Run", type = 'primary'): 
        data, labels, title = generate_data(option, numberDatapoints)
        data = pd.DataFrame(data, columns=["x","y"]) 

        st.session_state['data'] = data  # Store generated data in session state
        st.session_state['labels'] = labels  # Store generated labels in session state
        st.session_state['title'] = title  # Store title in session state

        # st.write(data)
        # st.scatter_chart(data, x="x", y="y")
        
    
    # Check if data is generated and stored in session state
    if 'data' in st.session_state:
        container.write("Chart here")
        container.write("### Select a Clustering Model and Run")
        data = st.session_state['data']
        title = st.session_state['title']
        labels = st.session_state['labels']
        st.scatter_chart(data, x="x", y="y")
        # Model selection dropdown

        container1 = st.container()
    
        sub_col1, sub_col2 = container1.columns(2)
        
        with sub_col1:
            model_option = st.selectbox("Choose clustering model", ("AGNES", "DIANA", "BIRCH", "Probabilistic"))
            
        with sub_col2:
            n_clusters = st.number_input(
                            "Number of clusters",
                            min_value=2,
                            ) 
        
        # Button to run clustering
        if st.button("Run Clustering"):
            # Apply clustering
            cluster_labels, model_data = apply_clustering(model_option, data, n_clusters)
            data_pd = pd.DataFrame(data, columns=["x", "y"])
            chart_area = st.empty()
            
            #Loop through the animation
            clusters = [[i] for i in range(numberDatapoints)]
            for i, (left, right) in enumerate(model_data.children_):
                new_cluster = clusters[left] + clusters[right]
                clusters.append(new_cluster)
                clusters[left] = []
                clusters[right] = []
                active_clusters = [c for c in clusters if c]
                labels = [find_cluster(item, active_clusters) for item in range(numberDatapoints)]
                data_pd["cluster"] = labels
                chart = alt.Chart(data_pd).mark_circle(size=60).encode(
                    x="x",
                    y="y",
                    color="cluster:N",
                    tooltip=["x", "y", "cluster"]
                    ).properties(
                        width=700,
                        height=500,
                        title=f"{title} - Clustering with {model_option}"
                ).interactive()
                
                chart_area.altair_chart(chart)
                if len(active_clusters) <= n_clusters:
                    break
                time.sleep(0.6)


if __name__ == "__main__":
    main()  
