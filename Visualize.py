import warnings

import altair as alt
import numpy as np
import streamlit as st
import pandas as pd

from sklearn.cluster import AgglomerativeClustering



warnings.filterwarnings("ignore")


def generate_data(x):
   return np.random.rand(x,2)


def main():
    st.title("Visualization clustering")
    container = st.container()
    
    col1, col2, col3 = container.columns(3)
    
    with col1:
        optionModel = st.selectbox(
        "Choosing model",
        ("Agglomerative", "BIRCH"),
        )
        
        
    with col2:
        numberClusters = st.number_input(
        "Number of cluster",
        min_value=0,
        )
        
            
    with col3:
        numberDatapoints = st.number_input(
        "Number of datapoint",
        min_value=2,
        ) 
        
    if container.button("Run", type = 'primary'):
        st.write("Chart here") 
        data= generate_data(numberDatapoints) 
        data_pd = pd.DataFrame(data, columns=["x","y"]) 
        st.scatter_chart(data_pd, x="x", y="y")    
    

if __name__ == "__main__":
    main()    
