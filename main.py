import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  

# Streamlit page config
st.set_page_config(
    page_title='Customer Segmentation',
    page_icon=':bar_chart:',
)


# Function to load the cluster results
def load_cluster_results():
    kmeans_df = joblib.load(r"kmeans_results.joblib") 
    gmm_df = joblib.load(r"kmeans_results.joblib")  
    birch_df = joblib.load(r"kmeans_results.joblib") 
    retail = joblib.load(r"kmeans_results.joblib")
    
    return kmeans_df, gmm_df, birch_df, retail


# Function to handle zeros or negatives
def handle_neg_n_zero(num):
    return max(num, 1)


def main():
    st.title("Customer Segmentation Dashboard")

    # Load datasets
    kmeans_df, gmm_df, birch_df, retail = load_cluster_results()

    # Sidebar model selector
    st.sidebar.header("Select a Clustering Model")
    model_choice = st.sidebar.selectbox("Model", ["KMeans", "GMM", "BIRCH"])

    # Use appropriate dataset
    if model_choice == "KMeans":
        df = kmeans_df.copy()
        cluster_col = "KMeans_Cluster"
    elif model_choice == "GMM":
        df = gmm_df.copy()
        cluster_col = "GMM_Cluster"
    else:
        df = birch_df.copy()
        cluster_col = "BIRCH_Cluster"

    # Show sample data
    st.markdown(f"### Sample Data ({model_choice})")
    st.dataframe(retail.head())

    # Show cluster visualization
    if "Recency" in df.columns and "Frequency" in df.columns:
        st.markdown(f"### Recency vs Frequency Clustering ({model_choice})")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x="Recency", y="Frequency", hue=cluster_col, palette="Set2", ax=ax)
        ax.set_title(f"{model_choice} Clusters")
        st.pyplot(fig)
    else:
        st.warning("Missing required columns for plotting.")

    # User RFM Input
    st.markdown("### Predict Your Cluster Based on RFM Input")

    today = datetime.today()
    recency_slider = st.slider(
        "Most Recent Purchase Date",
        min_value=today - timedelta(days=365),
        max_value=today,
        value=today - timedelta(days=30),
        format="DD/MM/YYYY"
    )
    recency_days = (today - recency_slider).days
    st.write(f"Recency: {recency_days} days")

    frequency_input = st.slider("Number of Visits", min_value=1, max_value=380, value=10)
    st.write(f"Frequency: {frequency_input}")

    monetary_input = st.slider("Total Spent", min_value=1.0, max_value=300000.0, value=100.0, step=10.0)
    st.write(f"Monetary: {monetary_input}")

    # Prepare data for prediction
    try:
        features = df[['Recency', 'Frequency', 'Monetary']]  # Use original features, not log-transformed
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        user_scaled = scaler.transform([[recency_days, frequency_input, monetary_input]])

        if model_choice == "KMeans":
            model = KMeans(n_clusters=df[cluster_col].nunique(), random_state=42)
            model.fit(scaled_features)
            prediction = model.predict(user_scaled)[0]
        elif model_choice == "GMM":
            model = GaussianMixture(n_components=df[cluster_col].nunique(), random_state=42)
            model.fit(scaled_features)
            prediction = model.predict(user_scaled)[0]
        else:  # BIRCH
            model = Birch(n_clusters=df[cluster_col].nunique())
            model.fit(scaled_features)
            prediction = model.predict(user_scaled)[0]

        st.success(f"You belong to Cluster: {prediction}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Run the app
if __name__ == "__main__":
    main()
