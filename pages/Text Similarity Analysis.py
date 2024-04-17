import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import io

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set page configuration
st.set_page_config(
    page_title="Text Analysis",
    page_icon=":memo:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to preprocess text
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return ' '.join(lemmatized_tokens)

# Function to calculate word count
def word_count(text):
    return len(text.split())

# Function to generate word cloud
def generate_wordcloud(text):
    if text:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt.gcf())
    else:
        st.write("No words to display.")

# Function to calculate cosine similarity
def cosine_similarity(text1, text2):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
    return sklearn_cosine_similarity(tfidf_matrix)[0, 1]

# Sidebar
st.sidebar.title('TEXT ANALYSIS OPTIONS')

# Main content
st.title("Text Analysis")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=['csv'])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write('### Dataset Summary')
    st.write(df.head())

    # Select rows
    row_selection = st.sidebar.multiselect('Select any two rows:', df['Review Text'])

    if len(row_selection) == 2:
        text1 = df[df['Review Text'] == row_selection[0]]['Review Text'].iloc[0]
        text2 = df[df['Review Text'] == row_selection[1]]['Review Text'].iloc[0]

        st.header('First Sentence Text Analysis')
        st.subheader("Word Count:")
        st.write(word_count(text1))
        cleaned_text1 = preprocess_text(text1)
        generate_wordcloud(cleaned_text1)

        st.header('Second Sentence Text Analysis')
        st.subheader("Word Count:")
        st.write(word_count(text2))
        cleaned_text2 = preprocess_text(text2)
        generate_wordcloud(cleaned_text2)

        st.header('Similarity Analysis of the Two Selected Sentences')
        cosine_sim = cosine_similarity(cleaned_text1, cleaned_text2)
        st.write('Cosine Similarity:', cosine_sim)

        # Plot similarity
        fig, ax = plt.subplots()
        similarity_labels = ['Cosine Similarity']
        similarity_scores = [cosine_sim]
        ax.bar(similarity_labels, similarity_scores, color=['blue'])
        ax.set_xlabel('Similarity Measures')
        ax.set_ylabel('Similarity Scores')
        ax.set_title('Similarity Analysis')
        st.pyplot(fig)
