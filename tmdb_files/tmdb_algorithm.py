from face_model_files import deepface_functions as df
from tmdb_files.tmdb_age_preprocessing import get_ages
from tmdb_files.step1_tmdb import convert_imdb_to_tmdb
from tmdb_files.step2_tmdb import get_approximate_movie_dates
from tmdb_files.step3_tmdb import calculate_release_range
from tmdb_files.media_ranking_alg import rank_media
from dotenv import load_dotenv
import os

def execute_program_on_image(img_path, save_path):
    load_dotenv()
    
    API_KEY = os.getenv("API_KEY")
    response = {}
    
    # ------- Models and Preprocessing -------
    
    faces_list = []

    # split faces and save to ../extracted_photos_temp
    faces = df.split_and_save(img_path, save_path)

    # Gets the IMDB_IDS and confidence values from the postgres_db
    imdb_objs = df.embed_and_search(faces, 5)
    imdb_ids = []
    for face_matches in imdb_objs:
        imdb_ids.append(face_matches[0].actor_id)

    # GetAges will be in the age_model_files directory and will use the photos that were saved to extracted_photos_temp and predict their ages
    ages = get_ages(save_path)

    # format response
    for i in range(len(ages)):
        new_face_obj = {"image": "n/a", "age": ages[i], "mae": 5.7}
        actors = []
        for cmp in imdb_objs[i]:
            new_actor_obj = {"name": cmp.name, "confidence": cmp.confidence}
            actors.append(new_actor_obj)
        new_face_obj["actors"] = actors
        faces_list.append(new_face_obj)

    response["faces"] = faces_list

    # ------- TMDB Algo -------

    # Convert from IMDB_ID to TMDB_ID
    tmdb_ids = convert_imdb_to_tmdb(imdb_ids, API_KEY)

    # Find the range of years of media we need to search through based on the algorithm we defined in Google Docs/Discord
    approximate_movie_dates = get_approximate_movie_dates(tmdb_ids, ages)
    approximate_year_range = calculate_release_range(approximate_movie_dates)
    
    # Find the possible ranges of media using the tmdb_ids and year range

    possible_media = rank_media(approximate_year_range, tmdb_ids, 5)
    response["media"] = possible_media
    print(response)

    return response
    

