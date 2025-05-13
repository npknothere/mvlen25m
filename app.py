from flask import Flask, render_template, request
from RecommendationSystem import RecommendationSystem
import pickle

with open("recommender.pkl", "rb") as f:
    recommender = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        title = request.form["title"]
        matches = recommender.search_movies(title)
        return render_template("results.html", movies=matches.to_dict(orient="records"),is_recommendation=False)
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    movie_id = int(request.form["movie_id"])
    recommendations = recommender.find_similar_movies(movie_id)
    return render_template("results.html", movies=recommendations.to_dict(orient="records"),is_recommendation=True)

if __name__ == "__main__":
    app.run(debug=True)