from uuid import NAMESPACE_URL

import requests
import streamlit as st
import pandas as pd
import numpy as np


# Recommendation function
def myIBCF(new_user_ratings):
    # Load top 30 similarity matrix data
    similarity_url = "https://raw.githubusercontent.com/hwangsamy1/CS598-PSL/refs/heads/main/Project4/similarity_matrix_top30.csv"
    similarity_matrix = pd.read_csv(similarity_url, index_col=0)
    predictions = {}

    # Predict the ratings for movies not rated by user
    for movie in similarity_matrix.index:
        if pd.isna(new_user_ratings[movie]):
            related_movies = similarity_matrix.loc[movie].dropna()
            rated_movies = new_user_ratings[~new_user_ratings.isna()]
            relevant_movies = related_movies.index.intersection(rated_movies.index)
            
            if relevant_movies.any():
                weights = related_movies.loc[relevant_movies]
                ratings = rated_movies.loc[relevant_movies]
                prediction = (weights * ratings).sum() / weights.sum()
                predictions[movie] = prediction
    
    # Sort by predicted ratings
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    predictions_top10_vals = np.array([val for movie, val in sorted_predictions[:10]])
    predictions_top10_movies = np.array([movie for movie, val in sorted_predictions[:10]])

    notna_count = np.count_nonzero(np.isnan(predictions_top10_vals))
    
    if predictions_top10_vals.size == 0:
        notna_count = 10
        
    if notna_count > 0:
        # Load popularity rankings
        popularity_ranks_url = "https://raw.githubusercontent.com/hwangsamy1/CS598-PSL/refs/heads/main/Project4/all_popular_ranking.csv"
        popularity_ranks = pd.read_csv(popularity_ranks_url, index_col=0)

        # Mask movies already in top predictions
        mask = ~np.isin(popularity_ranks, predictions_top10_movies)
        popular_noranked = popularity_ranks[mask]
        remaining_movies = popular_noranked[:notna_count].to_numpy().flatten()

        print(remaining_movies)
        # Merge predictions
        new_predictions = np.full(10, '', dtype='<U10')
        new_predictions[:len(predictions_top10_movies)] = predictions_top10_movies

        new_predictions[(10-notna_count):] = remaining_movies
        return new_predictions

    return predictions_top10_movies


# Main App
def main():
    st.set_page_config(layout="wide")
    st.title("Movie Recommender System")
    st.subheader("Rate Movies to Get Personalized Recommendations")

    # Show sample movies to rate
    movie_names_url = "https://github.com/hwangsamy1/CS598-PSL/raw/refs/heads/main/Project4/data/movies.dat"
    response = requests.get(movie_names_url)

    # Split the data into lines and then split each line using "::"
    movie_lines = response.text.split('\n')
    movie_data = [line.split("::") for line in movie_lines if line]

    # Create a DataFrame from the movie data
    movies_data = pd.DataFrame(movie_data, columns=['movie_id', 'title', 'genres'])
    movies_data['movie_id'] = movies_data['movie_id'].astype(int)
    movies = movies_data.iloc[:100, :]

    movie_list_url = "https://raw.githubusercontent.com/hwangsamy1/CS598-PSL/refs/heads/main/Project4/all_popular_ranking.csv"
    movies_list = pd.read_csv(movie_list_url, index_col=0).to_numpy()

    img_url = 'https://raw.githubusercontent.com/hwangsamy1/CS598-PSL/refs/heads/main/Project4/MovieImages/'

    st.write("Please rate the following sample movies (1-5 stars or leave blank):")
    user_ratings = {key[0]: np.nan for key in movies_list}

    radio_dict = {
        'None': pd.NA,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5
    }

    cols = st.columns(4)

    for index, movie in movies.iterrows():
        cols[index % 4].image(img_url + str(movie['movie_id']) + '.jpg')
        rating = cols[index % 4].radio(f"{movie['title']}", ['None', '1', '2', '3', '4', '5'], horizontal=True)
        movie_id = ("m" + str(movie['movie_id'])).strip().encode('utf-8').decode('utf-8')
        user_ratings[movie_id] = radio_dict.get(rating)

    # Convert user ratings to a Pandas Series
    user_ratings_series = pd.Series(user_ratings)

    # Show recommendations if ratings are provided
    if st.button("Get Recommendations"):
        if user_ratings_series.empty:
            st.warning("Please rate at least one movie to get recommendations.")
        else:
            recommendations = myIBCF(user_ratings_series)
            st.success("Top 10 Recommended Movies for You:")

            for movie in recommendations:
                movie_id = int(movie[1:])
                result = movies_data.loc[movies_data['movie_id'] == movie_id]['title'].values[0]
                st.write(result)


if __name__ == "__main__":
    main()
