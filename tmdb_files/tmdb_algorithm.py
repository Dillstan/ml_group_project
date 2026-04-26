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

    # ------- Models and Preprocessing -------
    
    #split faces and save to ../extracted_photos_temp
    faces = df.split_and_save(img_path, save_path)

    #Gets the IMDB_IDS and confidence values from the postgres_db
    imdb_objs = df.embed_and_search(faces)
    imdb_ids = []
    for face_matches in imdb_objs:
        for cmp in face_matches:
            imdb_ids.append(cmp.actor_id)

    #GetAges will be in the age_model_files directory and will use the photos that were saved to extracted_photos_temp and predict their ages
    ages = get_ages(save_path)

    
    # ------- TMDB Algo -------

    #Convert from IMDB_ID to TMDB_ID
    tmdb_ids = convert_imdb_to_tmdb(imdb_ids, API_KEY)
    # print(tmdb_ids)

    #use the TMDB_ID to get all info we can on the actor (dob, name are most important here)
    # actor_info = GetActorInfo(tmdb_ids)

    #Find the range of years of media we need to search through based on the algorithm we defined in Google Docs/Discord
    approximate_movie_dates = get_approximate_movie_dates(tmdb_ids, ages)
    approximate_year_range = calculate_release_range(approximate_movie_dates)

    #Find the possible ranges of media using the tmdb_ids and year range
    # Get Media will also be where Max's RankMedia() function is called.
    # year_range = (1994,2008)
    # tmdb_ids = [20810, 20387, 57581, 887, 20810]
    possible_media = rank_media(approximate_year_range, tmdb_ids, 5)
    print(possible_media)

    #this is what we return to the UI for displaying
    #return possible_media, actor_info, imdb_ids

