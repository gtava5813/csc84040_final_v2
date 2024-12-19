import streamlit as st
import numpy as np
from apps_utils import SongRecommender


def format_recommendations(recommendations):
    formatted = []
    for rank, (item, score) in enumerate(recommendations, start=1):
        formatted.append(f"{rank}. {item} - Score: {score:.2f}")
    return formatted



def recommend_hybrid(tfidf_recommendations,
                     encoder_recommendations,
                     cf_recommendations ,
                     top_k=5,  
                     weights = {'tfidf': 0.3,
                                'encoder': 0.3,
                                'collaborative': 0.4}): # weights to each model (can be tuned)
    
    # Normalize similarity scores to 0-1 range
    def normalize_scores(recommendations):
        scores = np.array([rec[1] for rec in recommendations])
        min_score = scores.min()
        max_score = scores.max()
        normalized = (scores - min_score) / (max_score - min_score)
        return [(rec[0], norm_score) for rec, norm_score in zip(recommendations, normalized)]

    # Combine normalized scores
    combined_scores = {}
    for item, score in normalize_scores(tfidf_recommendations):
        combined_scores[item] = weights['tfidf'] * score
    
    for item, score in normalize_scores(encoder_recommendations):
        combined_scores[item] = combined_scores.get(item, 0) + weights['encoder'] * score
        
    for item, score in normalize_scores(cf_recommendations):
        combined_scores[item] = combined_scores.get(item, 0) + weights['collaborative'] * score
    
    # Sort and return top K recommendations
    sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return format_recommendations(sorted_items[:top_k])



st.title("Song Recommender App")

# Initialize recommender
@st.cache_resource
def load_recommender():
    recommender = SongRecommender()
    recommender.load_lyrics_data("lyrics_dataset/csv")
    recommender.build_tfidf_model()
    recommender.build_encoder_model()
    recommender.load_spotify_data('spotify_data.csv')
    return recommender

recommender = load_recommender()

# User input
query_song = st.text_input("Enter a song (Artist - Title):","Coldplay - The Scientist")
if st.button("Get Recommendations"):
    # Get recommendations using different methods
    tfidf_recommendations = recommender.recommend_by_tfidf(query_song)
    encoder_recommendations = recommender.recommend_by_encoder(query_song)
    cf_recommendations = recommender.recommend_by_collaborative_filtering(query_song)
    final_recommendations = recommend_hybrid(tfidf_recommendations, encoder_recommendations, cf_recommendations)

        # Display results
    st.header("TF-IDF Recommendations")
    for song, score in tfidf_recommendations:
        st.write(f"- {song} (similarity: {score:.2f})")

    st.header("Encoder Recommendations")
    for song, score in encoder_recommendations:
        st.write(f"- {song} (similarity: {score:.2f})")

    st.header("Collaborative Filtering Recommendations")
        for song, score in cf_recommendations:
        st.write(f"- {song} (score: {score:.4f})")
            
            
    st.header("\nCombined Normalized and Weighted Recommendations:")    
    for recommendation in final_recommendations:
        st.write(recommendation)
