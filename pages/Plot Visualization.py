import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Page layout
st.title('3D Plot Visualization')

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Load and display dataset summary
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write('### Dataset Summary')
    st.write(data.head())

    # Data preprocessing
    numeric_data = data[['Age', 'Rating', 'Positive Feedback Count']].dropna()
    X = numeric_data.values
    X = StandardScaler().fit_transform(X)

    # Plot 3D scatter plot
    st.write('### 3D Scatter Plot')
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection='3d')

    # Define colors for each column
    colors = ['r', 'g', 'b']

    for i in range(X.shape[1]):
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, i], cmap=plt.cm.viridis, label=numeric_data.columns[i])

    ax.set_xlabel('Age')
    ax.set_ylabel('Rating')
    ax.set_zlabel('Positive Feedback Count')
    ax.legend()
    st.pyplot(fig)

    # Additional insights
    st.write('### Additional Insights')
    st.write('- The 3D scatter plot visualizes the relationship between Age, Rating, and Positive Feedback Count columns.')
    st.write('- Each point represents a data point from the dataset, with different colors representing different columns.')
    st.write('- Users can explore the plot to identify any patterns or clusters in the data.')
