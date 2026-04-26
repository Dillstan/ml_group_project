# Step 2 - Get actor age during filming of scene
# April 25th, 2026

import requests
import os
from dotenv import load_dotenv

def get_approximate_movie_dates(tmdb_ids, ages):

    load_dotenv() # load environment variables
    tmdb_access_token = os.getenv("TMDB_BEARER")

    years_of_filming = []

    for tmdb_id, age in zip(tmdb_ids, ages):

        url = f"https://api.themoviedb.org/3/person/{tmdb_id}?language=en-US"

        headers = {
            'Authorization': f"Bearer {tmdb_access_token}",
            'accept': 'application/json'
        }

        response = requests.request("GET", url, headers=headers)
        
        if (response.status_code == 200):
            data = response.json()
            if (data.get('birthday')):
                year_of_movie = int(data.get('birthday').split("-")[0]) + age
                years_of_filming.append(year_of_movie)

        else:
            print(f"API Error for {tmdb_id}: {response.status_code}")
            years_of_filming.append(None)

    return years_of_filming