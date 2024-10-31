# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import streamlit as st

# Load the dataset
df = pd.read_csv(r"C:\Users\sunil\Desktop\DK\NIT\NIT- Data Science and AI Class\3. October\28th, 29th - clustering\2.K-MEANS CLUSTERING\Mall_Customers.csv")
data = df.iloc[:, [3, 4]].values  # Select relevant features

# Function to predict cluster
def predict_cluster(income, spending):
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
    kmeans.fit(data)  # Fit model on the data
    input_data = np.array([[income, spending]])
    cluster = kmeans.predict(input_data)  # Predict the cluster
    return cluster[0]  # Return the cluster number


st.title("Cluster Compass APP")

name=st.text_input("Enter Your name")

# User inputs for Annual Income and Spending Score
annual_income = st.number_input("Enter Annual Income ($k)", min_value=0, max_value=200, value=0)
spending_score = st.number_input("Enter Spending Score (1-100)", min_value=1, max_value=100, value=1)



# Button to predict cluster
if st.button("Predict Cluster"):
    cluster_result = predict_cluster(annual_income, spending_score)
    st.write(f"Mr/ Mrs {name} you belongs to Cluster {cluster_result + 1}!")

    # Visualize the clusters with input data
    plt.figure(figsize=(10, 6))
    y_kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0).fit_predict(data)
    
    # Scatter plot for clusters
    for i in range(5):
        plt.scatter(data[y_kmeans == i, 0], data[y_kmeans == i, 1], s=100, label=f'Cluster {i+1}')
        
    # Highlight input point
    plt.scatter(annual_income, spending_score, s=200, c='orange', label='Input Data Point', edgecolor='black')

    # Plotting centroids
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
    kmeans.fit(data)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', label='Centroids')
    
    plt.title('K-Means Clustering of Customers')
    plt.xlabel('Annual Income ($k)')
    plt.ylabel('Spending Score (1 - 100)')
    plt.legend()
    
    st.pyplot(plt)  # Display the plot in Streamlit
