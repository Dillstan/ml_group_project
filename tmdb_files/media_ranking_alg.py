# Media Ranking Algorithm - Maxwell Klema
# April 21st, 2026

import requests
import concurrent.futures
import json
import os
from dotenv import load_dotenv

load_dotenv() # load environment variables

# make request to get list of movie credits for each actor
def fetch_movies(tmdb_id):
    tmdb_access_token = os.getenv("TMDB_BEARER")
    url = f'https://api.themoviedb.org/3/person/{tmdb_id}/movie_credits?language=en-US'
    headers = {
  'Authorization': f'Bearer {tmdb_access_token}',
  'accept': 'application/json'
    }

    response = requests.request("GET", url, headers=headers)
    json_response = json.loads(response.text)
    return json_response["cast"]

def rank_media(year_range, tmdb_ids, num_places):

    # fetch movie credits for each actor
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(fetch_movies, tmdb_ids))

    # calculate rankings

    rankings = []



    return rankings


# testing
year_range = (1980,1990)
tmdb_ids = [521, 1064, 44735]
num_places = 5

print(rank_media(year_range, tmdb_ids, num_places))
# with open("ranking.json", "w") as f:
#     rankings = rank_media(year_range, tmdb_ids, num_places)
#     f.write(str(rankings))