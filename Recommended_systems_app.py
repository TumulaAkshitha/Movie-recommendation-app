import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv("C:\\Users\\tumul\\Downloads\\IMDB Movie Data.csv")
    return data

# Function to get movie recommendations
def get_recommendations(data, movie_name):
    tf = TfidfVectorizer()
    tfidf_matrix = tf.fit_transform(data['Movie_Genre'])
    
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    idx = data[data['Movie_name'] == movie_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return data['Movie_name'].iloc[movie_indices]

# Streamlit app
def main():
    st.title('Movie Recommendation System')
    
    data = load_data()
    
    movie_name = st.text_input("Enter a Movie Name")
    
    if st.button("Get Recommendations"):
        if movie_name:
            recommendations = get_recommendations(data, movie_name)
            st.markdown(f"**Recommended movies for '{movie_name}':**")
            st.write(recommendations)
        else:
            st.warning("Please enter a movie name.")

if __name__ == '__main__':
    main()
