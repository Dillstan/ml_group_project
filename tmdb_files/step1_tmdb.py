import requests

def convert_imdb_to_tmdb(imdb_ids, tmdb_api_key):
    """
    Takes an array of IMDb IDs and returns an array of TMDb IDs.
    """
    tmdb_ids = []
    
    for imdb_id in imdb_ids:
        url = f"https://api.themoviedb.org/3/find/{imdb_id}?api_key={tmdb_api_key}&external_source=imdb_id"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('person_results') and len(data['person_results']) > 0:
                tmdb_ids.append(data['person_results'][0]['id'])
            else:
                tmdb_ids.append(None)
        else:
            print(f"API Error for {imdb_id}: {response.status_code}")
            tmdb_ids.append(None)
            
    return tmdb_ids