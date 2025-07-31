import streamlit as st
import pandas as pd
import joblib

# Load models and data
kmeans = joblib.load("models/kmeans_rfm_model.pkl")
scaler = joblib.load("models/rfm_scaler.pkl")
product_sim_df = pd.read_pickle("models/product_similarity.pkl")

# UI Configuration
st.set_page_config(page_title="Shopper Spectrum", layout="wide")
st.title("🛒 Shopper Spectrum Dashboard")

tab1, tab2 = st.tabs(["📦 Product Recommender", "👤 Customer Segmentation"])

with tab1:
    st.header("🔍 Product Recommendation System")
    product_id = st.text_input("Enter a Product ID (StockCode):")
    if st.button("Get Recommendations"):
        if product_id in product_sim_df.index:
            recommendations = product_sim_df[product_id].sort_values(ascending=False).drop(product_id).head(5)
            st.success("Top 5 Recommended Products:")
            st.write(recommendations)
        else:
            st.warning("Product not found in the dataset.")

with tab2:
    st.header("📊 Customer RFM Segmentation")
    recency = st.number_input("Recency (days)", min_value=0)
    frequency = st.number_input("Frequency (transactions)", min_value=0)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0)

    if st.button("Predict Cluster"):
        input_data = scaler.transform([[recency, frequency, monetary]])
        cluster = kmeans.predict(input_data)[0]
        avg_monetary = kmeans.cluster_centers_[:, 2]
        rank = avg_monetary.argsort()[::-1].tolist().index(cluster)
        label = ["High-Value", "Regular", "Occasional", "At-Risk"][rank]
        st.success(f"Predicted Segment: {label}")
