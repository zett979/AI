import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

if "uploaded_data" in st.session_state:
    df = st.session_state.uploaded_data

    st.subheader("Data Preview")
    st.write(df.head())

    # column names for clustering
    st.subheader("Select Columns for Clustering")
    # only select number column
    columns = df.select_dtypes(include=[np.number]).columns
    selected_columns = st.multiselect("Select numeric columns for clustering", columns)

    if selected_columns:
        data = df[selected_columns].dropna()

        st.subheader("Standardising the Data")
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        k_range = st.slider("Choose the range of clusters", 2, 10, (2, 6))
        k_min, k_max = k_range

        silhouette_scores = []
        for k in range(k_min, k_max + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data_scaled)
            score = silhouette_score(data_scaled, kmeans.labels_)
            silhouette_scores.append(score)

        # Plot the silhouette scores
        st.subheader("Silhouette Scores for Different Number of Clusters")
        fig, ax = plt.subplots()
        ax.plot(range(k_min, k_max + 1), silhouette_scores)
        ax.set_title("Silhouette Scores for K-Means Clustering")
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Silhouette Score")
        st.pyplot(fig)

        # Show the best number of clusters based on the highest silhouette score
        best_k = range(k_min, k_max + 1)[np.argmax(silhouette_scores)]
        st.write(
            f"Best value for k is {best_k} (Highest Silhouette Score: {max(silhouette_scores):.3f})"
        )

        k = st.slider("Select the number of clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_scaled)

        st.subheader("Elbow Method for Optimal Clusters")
        data = []
        for i in range(1, 11):
            kmeans_temp = KMeans(n_clusters=i, random_state=42)
            kmeans_temp.fit(data_scaled)
            data.append(kmeans_temp.inertia_)

        fig, ax = plt.subplots()
        ax.plot(range(1, 11), data, marker="o")
        ax.set_title("Elbow Method For Optimal Clusters")
        ax.set_xlabel("Number of Clusters")
        st.pyplot(fig)

else:
    st.write("Please upload a CSV file to get started.")
