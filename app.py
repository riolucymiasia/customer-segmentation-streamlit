import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load saved clustering model
with open("agg_clustering_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Agglomerative Clustering - Customer Segmentation App")
st.write("Upload your customer data to apply clustering based on average linkage.")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.write(df.head())

    try:
# Encode categorical columns
        for col in df.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

# Distance matrix & linkage (average method)
        distance_matrix = pdist(df)
        linkage_matrix = linkage(distance_matrix, method='average')

# Assign clusters using the number from the saved model
        labels = fcluster(linkage_matrix, t=model.n_clusters, criterion='maxclust')
        df['Cluster'] = labels

# Sidebar filter
        st.sidebar.subheader("🔍 Filter by Cluster")
        unique_clusters = sorted(df['Cluster'].unique())
        selected_clusters = st.sidebar.multiselect("Select cluster(s) to view", unique_clusters, default=unique_clusters)
        filtered_df = df[df['Cluster'].isin(selected_clusters)]

# Show filtered data
        st.subheader("📂 Filtered Clustered Data")
        st.write(filtered_df.head())

# Cluster distribution
        st.subheader("📊 Cluster Distribution")
        st.bar_chart(df['Cluster'].value_counts())

# PCA-based visualization
        st.subheader("Cluster Visualization (PCA)")
        try:
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(df.drop('Cluster', axis=1))
            reduced_df = pd.DataFrame(reduced_data, columns=['PCA1', 'PCA2'])
            reduced_df['Cluster'] = df['Cluster'].values

            fig, ax = plt.subplots()
            sns.scatterplot(data=reduced_df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=80, ax=ax)
            ax.set_title("Customer Segments Visualized")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"PCA visualization failed: {e}")

# Download option
        csv = df.to_csv(index=False)
        st.download_button("Download Clustered CSV", csv, "clustered_data.csv", "text/csv")

    except Exception as e:
        st.error(f" An error occurred during clustering: {e}")

else:
    st.info("Please upload a dataset in CSV format.")