import os
import regex as re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

class SongRecommender:
    def __init__(self):
        self.lyrics_df = None
        self.tfidf_similarity_matrix = None
        self.encoder_embeddings = None
        self.songs_and_artists = None
        self.spotify_df = None
        self.spotify_similarity_matrix = None
        self.stop_words = self._get_stop_words()
        self.drop_words = self._get_drop_words()
        
    def _get_drop_words(self):
        drop_words = ["remix", "mix)", "mix]", "(live", "[live", "live from", "recorded live", "version)",
                      "version]", "edit)", "edit]", "edited)", "edited]", "demo)", "demo]", "the beyonce experience live",
                      "homecoming live", "(acoustic", "[acoustic", "acoustic)", "acoustic]"]
        return '|'.join([re.escape(word) for word in drop_words])
    
    def _get_stop_words(self):
        return ["a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also", "although",
                "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere",
                "are", "around", "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
                "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt",
                "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough",
                "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for",
                "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her",
                "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
                "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill",
                "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody",
                "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours",
                "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several",
                "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such",
                "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these",
                "they", "thick", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty",
                "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas",
                "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would",
                "yet", "you", "your", "yours", "yourself", "yourselves", "remix", "mix", "produced", "producer", "edit", "oh", "ah" "ra", "la", "\u200b"]

    def load_lyrics_data(self, data_folder):
        """
        Load and preprocess lyrics data from CSV files
        
        Args:
            data_folder (str): Path to folder containing CSV files
        Returns:
            self for method chaining
        """
        lyrics_df = pd.DataFrame()
        
        for file in os.listdir(data_folder):
            if file.endswith(".csv"):
                # Read CSV file
                df = pd.read_csv(os.path.join(data_folder, file))
                
                # Basic preprocessing
                if 'Lyric' in df.columns and 'Artist' in df.columns and 'Title' in df.columns:
                    # Drop rows with missing lyrics
                    df = df.dropna(subset=["Lyric"])
                    
                    # Create combined title and artist column
                    df["Title and Artist"] = df["Artist"] + " - " + df["Title"]
                    
                    # Select only needed columns
                    df = df[["Title and Artist", "Lyric"]]
                    
                    # Append to master dataframe
                    lyrics_df = pd.concat([lyrics_df, df], ignore_index=True)
        
        if lyrics_df.empty:
            raise ValueError("No valid lyrics data found in the specified folder")
            
        self.lyrics_df = lyrics_df
        self.songs_and_artists = lyrics_df['Title and Artist'].tolist()
        return self

    def _preprocess_lyrics_df(self, df):
        df = df.dropna(subset=["Lyric"])
        df = df[~df["Title"].str.contains(self.drop_words, case=False, na=False)]
        df["Title and Artist"] = df["Artist"] + " - " + df["Title"]
        return df[["Title and Artist", "Lyric"]]

    def build_tfidf_model(self):
        tfidf = TfidfVectorizer(
            max_features=None,
            stop_words=self.stop_words,
            lowercase=True
        )
        tfidf_matrix = tfidf.fit_transform(self.lyrics_df['Lyric'])
        self.tfidf_similarity_matrix = cosine_similarity(tfidf_matrix)
        return self

    def build_encoder_model(self, model_name='all-MiniLM-L6-v2'):
        model = SentenceTransformer(model_name)
        lyrics = self.lyrics_df['Lyric'].tolist()
        self.encoder_embeddings = model.encode(lyrics, convert_to_tensor=True)
        return self

    def load_spotify_data(self, file_path):
        features = ['danceability', 'energy', 'key', 'loudness', 'mode',
                   'speechiness', 'acousticness', 'instrumentalness',
                   'liveness', 'valence', 'tempo']
        self.spotify_df = pd.read_csv(file_path)[['user_id', 'artistname', 'trackname'] + features]
        self.spotify_similarity_matrix = cosine_similarity(self.spotify_df.iloc[:, 3:].values)
        return self

    def recommend_by_tfidf(self, query_song, top_n=5):
        try:
            idx = self.songs_and_artists.index(query_song)
        except ValueError:
            return f"Song '{query_song}' not found in the dataset."
        
        similarity_scores = list(enumerate(self.tfidf_similarity_matrix[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = [s for s in similarity_scores if s[0] != idx]
        return [(self.songs_and_artists[i], score) for i, score in similarity_scores[:top_n]]

    def recommend_by_encoder(self, query_song, top_n=5):
        try:
            idx = self.songs_and_artists.index(query_song)
        except ValueError:
            return f"Song '{query_song}' not found in the dataset."
            
        query_embedding = self.encoder_embeddings[idx]
        similarity_scores = util.pytorch_cos_sim(query_embedding, self.encoder_embeddings)[0]
        top_indices = similarity_scores.argsort(descending=True)[1:top_n+1]
        return [(self.songs_and_artists[i], similarity_scores[i].item()) for i in top_indices]

    def recommend_by_collaborative_filtering(self, query_song, n=5):
        artist, track = query_song.split(" - ", 1)
        
        target_song_index = self.spotify_df[
            (self.spotify_df['artistname'] == artist) & 
            (self.spotify_df['trackname'] == track)
        ].index.tolist()
        
        if not target_song_index:
            print(f"Song '{query_song}' not found in the dataset.")
            return []
        
        target_song_index = target_song_index[0]
        similarity = self.spotify_similarity_matrix[target_song_index]
        
        # Create a list of tuples with index, artist, track, and similarity score
        recommendations = []
        seen_songs = set()
        
        for idx, score in enumerate(similarity):
            song_key = (self.spotify_df.iloc[idx]['artistname'], 
                    self.spotify_df.iloc[idx]['trackname'])
            
            if (idx != target_song_index and 
                song_key not in seen_songs):
                recommendations.append((idx, *song_key, float(score)))
                seen_songs.add(song_key)
        
        recommendations.sort(key=lambda x: x[3], reverse=True)
        return [(artist+" - "+track, score) 
                for _, artist, track, score in recommendations[:n]]


