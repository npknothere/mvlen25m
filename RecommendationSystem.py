import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

class RecommendationSystem:
    def __init__(self, movies_df, ratings_df,vectorizer, tfidf):
        self.vectorizer = vectorizer
        self.tfidf = tfidf
        self.movies = movies_df
        self.ratings = ratings_df

    def clean_text(self,text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text

    def search_movies(self,title):
        title = self.clean_text(title)
        user_title = self.vectorizer.transform([title])
        similarity = cosine_similarity(user_title, self.tfidf).flatten()
        index = np.argsort(similarity)[-10:][::-1]
        result = self.movies.iloc[index]
        return result
    
    def find_similar_movies(self,movie_id):
        similar_users = self.ratings[(self.ratings["movieId"] == movie_id) & (self.ratings["rating"] > 4)]["userId"].unique()
        similar_user_recs = self.ratings[(self.ratings["userId"].isin(similar_users)) & (self.ratings["rating"] > 4)]["movieId"]
        similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

        similar_user_recs = similar_user_recs[similar_user_recs > .10]
        all_users = self.ratings[(self.ratings["movieId"].isin(similar_user_recs.index)) & (self.ratings["rating"] > 4)]
        all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
        rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
        rec_percentages.columns = ["similar", "all"]
        
        rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
        rec_percentages = rec_percentages.sort_values("score", ascending=False)
        return rec_percentages.head(10).merge(self.movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]