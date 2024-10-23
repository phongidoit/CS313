import warnings

import altair as alt
import numpy as np
import streamlit as st

from sklearn.cluster import AgglomerativeClustering



warnings.filterwarnings("ignore")


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
        min_value=0,
        )    
    

if __name__ == "__main__":
    main()    
