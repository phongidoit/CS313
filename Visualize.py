import warnings

import altair as alt
import numpy as np
import streamlit as st
import pandas as pd

from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering


warnings.filterwarnings("ignore")


def generate_data(x):
   return np.random.rand(x,2)

def generate_data2(dataset_name, n_samples=50, seed=30):
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
        random_state = 170
        X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_aniso = np.dot(X, transformation)
        data = (X_aniso, y)
        title = "Anisotropic Blobs"
    elif dataset_name == "varied":
        random_state = 170
        data = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
        title = "Blobs with Varied Variances"
    else:
        raise ValueError("Unknown dataset name.")
    return np.array(data[0]), np.array(data[1]), title


def main():
    st.title("Visualization clustering")
    container = st.container()
    
    col1, col2, col3 = container.columns(3)
    data = None
    
    with col1:
        optionModel = st.selectbox(
        "Choosing model",
        ("Agglomerative", "BIRCH"),
        )
        
        
    with col2:
        # numberClusters = st.number_input(
        # "Number of cluster",
        # min_value=0,
        # )
        option = st.selectbox("Choose data type?",("noisy_circles", "noisy_moons", "blobs", "no_structure", "aniso", ),)
        
            
    with col3:
        numberDatapoints = st.number_input(
        "Number of datapoint",
        min_value=2,
        ) 
        
    if container.button("Run", type = 'primary'):
        st.write("Chart here") 
        # data= generate_data(numberDatapoints) 
        data, labels, title = generate_data2(option, numberDatapoints)
        st.write(data)
        data_pd = pd.DataFrame(data, columns=["x","y"]) 
        st.scatter_chart(data_pd, x="x", y="y")    
    

if __name__ == "__main__":
    main()    
