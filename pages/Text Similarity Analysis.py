import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Page layout
st.title('Text Similarity Analysis')

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Load and display dataset summary
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write('### Dataset Summary')
    st.write(data.head())

    # Text processing
    text_data = data['Review Text'].dropna()

    # Tokenization, remove stopwords, punctuation, special characters, and convert to lowercase
    stop_words = set(stopwords.words('english'))
    punctuations = set(string.punctuation)
    processed_text = []

    for text in text_data:
        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalpha()]
        tokens = [word for word in tokens if word not in stop_words and word not in punctuations]
        processed_text.append(' '.join(tokens))

    # Stemming or Lemmatization
    option = st.radio('Select text normalization technique:', ('Stemming', 'Lemmatization'))

    if option == 'Stemming':
        stemmer = PorterStemmer()
        processed_text = [' '.join([stemmer.stem(word) for word in text.split()]) for text in processed_text]
    else:
        lemmatizer = WordNetLemmatizer()
        processed_text = [' '.join([lemmatizer.lemmatize(word) for word in text.split()]) for text in processed_text]

    # Text similarity analysis
    division_names = ['General', 'General Petite', 'Initmates']
    selected_division = st.selectbox('Select a division:', division_names)

    # Filter data based on selected division
    selected_data = data[data['Division Name'] == selected_division]
    text_data_selected = selected_data['Review Text'].dropna()

    # Get indices of selected data
    selected_indices = text_data_selected.index

    # Check if selected indices are empty
    if len(selected_indices) > 0:
        # Filter processed text based on selected indices
        processed_text_selected = [processed_text[i] for i in selected_indices]

        # Calculate TF-IDF matrix
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(processed_text_selected)

        # Compute cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Display similar reviews
        st.write('### Similar Reviews within', selected_division, 'Division:')
        for i, row in enumerate(similarity_matrix):
            similar_indices = np.where(row > 0.7)[0]  # Threshold for similarity
            if len(similar_indices) > 1:
                st.write('Review:', selected_indices[i], '-', text_data_selected.iloc[i])
                for idx in similar_indices:
                    if idx != i:
                        st.write('Similar Review:', selected_indices[idx], '-', text_data_selected.iloc[idx])
                st.write('---')
    else:
        st.write('No data available for the selected division.')
