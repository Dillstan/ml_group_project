# Media Ranking Algorithm - Maxwell Klema
# April 21st, 2026

import requests
import concurrent.futures
import json
import os
from dotenv import load_dotenv
from datetime import date
from collections import defaultdict
from itertools import repeat

load_dotenv() # load environment variables
rankings_dict = {}

# make request to get list of movie credits for each actor
def fetch_movies(tmdb_id):
    tmdb_access_token = os.getenv("TMDB_BEARER")
    url = f'https://api.themoviedb.org/3/person/{tmdb_id}/movie_credits?language=en-US'
    headers = {
  'Authorization': f'Bearer {tmdb_access_token}',
  'accept': 'application/json'
    }

    response = requests.request("GET", url, headers=headers)
    if response.status_code == 200:
        json_response = json.loads(response.text)
        return json_response["cast"]
    else:
        print(f"API Error for {tmdb_id}: {response.status_code}")
        return []

def update_movie_frequency(movie_credit, year_range):
    for movie in movie_credit: 
        if (len(movie["release_date"]) == 0): # exclude in-valid movie dates
            continue
        
        movie_date_split = movie["release_date"].split("-")
        movie_year = int(movie_date_split[0])
        movie_month = int(movie_date_split[1])
        movie_day = int(movie_date_split[2])
        if (movie_year <= year_range[1] and movie_year >= year_range[0]):
            movie_details = (movie["title"], date(movie_year, movie_month, movie_day), movie["poster_path"], movie["id"]) # movie title, release date, poster img path
            rankings_dict[movie_details] = rankings_dict.get(movie_details, 0) + 1 # increase frequency count of movie

# grab IMDB ID
def get_imdb_id(internal_id, type="movie"):
    tmdb_access_token = os.getenv("TMDB_BEARER")
    url = f'https://api.themoviedb.org/3/{type}/{internal_id}/external_ids'
    headers = {
        'Authorization': f'Bearer {tmdb_access_token}',
        'accept': 'application/json'
    }
    response = requests.request("GET", url, headers=headers)
    if response.status_code == 200:
        json_response = json.loads(response.text)
        return json_response["imdb_id"]
    else:
        print(f"API Error for {internal_id}: {response.status_code}")
        return "N/A"
        
# format as JSON list
def format_json(results):
    media = []
    for result in results:
        imdb_id = get_imdb_id(result[3])
        movie_obj = {"title": result[0], "year": result[1].year, "poster": f"https://image.tmdb.org/t/p/w500{result[2]}", "imdbId": imdb_id}
        media.append(movie_obj)
    return media

def rank_media(year_range, tmdb_ids, num_places=5):
    rankings_dict.clear()
    tmdb_ids = set(tmdb_ids)
    tmdb_ids = [x for x in tmdb_ids if x is not None]
    print(tmdb_ids)
    # fetch movie credits for each actor
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_movies, tmdb_ids))

    # calculate rankings
    rankings = []

    # iterate over all movies to map frequencies
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(update_movie_frequency, results, repeat(year_range))

    # group movies by frequencies
    grouped_by_freq = defaultdict(set)

    for movie, freq in rankings_dict.items():
        grouped_by_freq[freq].add(movie)
    
    grouped_by_freq = dict(grouped_by_freq)

    # for each frequency count, order by closest to median age in tuple
    median_date = int((year_range[0] + year_range[1]) / 2)
    if (median_date % 2 == 0):
        median_date = date(median_date, 1, 1)
    else:
        median_date = date(median_date, 7, 2) # middle day of the year

    # for each frequency, append movies in order of them being closest to the median date

    for freq, movie_set in sorted(grouped_by_freq.items(), reverse=True):
        freq_rankings = []
        for movie_tuple in movie_set:
            age_differences = abs(median_date - movie_tuple[1])
            freq_rankings.append((age_differences, movie_tuple))
        freq_rankings.sort()
        rankings.extend(freq_rankings)

    rankings_array = [movie[1] for movie in rankings[:num_places]]
    resp = format_json(rankings_array)
    return resp