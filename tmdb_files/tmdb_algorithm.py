from face_model_files import deepface_functions as df
from tmdb_files.tmdb_age_preprocessing import get_ages
from tmdb_files.step1_tmdb import convert_imdb_to_tmdb
from tmdb_files.step2_tmdb import get_approximate_movie_dates
from tmdb_files.step3_tmdb import calculate_release_range
from tmdb_files.media_ranking_alg import rank_media
from dotenv import load_dotenv
import base64
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
    imdb_objs = df.embed_and_search(faces, top_n_faces=3)
    imdb_ids = []
    for face_matches in imdb_objs:
        if len(face_matches) > 0:
            imdb_ids.append(face_matches[0].actor_id)
        else:
            imdb_ids.append(None)

    # GetAges will be in the age_model_files directory and will use the photos that were saved to extracted_photos_temp and predict their ages
    ages = get_ages(save_path)

    # format response
    for i in range(len(ages)):
        # get base64 of image
        with open(f"~extracted_photos_temp/{i}.jpg", "rb") as img_file:
            b64_bytes = base64.b64encode(img_file.read())
            b64_string = b64_bytes.decode("utf-8")
        
        new_face_obj = {"image": f"data:image/jpg;base64,{b64_string}", "age": ages[i], "mae": 5.7}
        actors = []
        for cmp in imdb_objs[i]:
            if (cmp != None):
                new_actor_obj = {"name": cmp.name, "confidence": cmp.confidence}
                actors.append(new_actor_obj)
            else:
                new_actor_obj = {"name": "Unknown", "confidence": "Very Low Confidence"}
                actors.append(new_actor_obj)
        new_face_obj["actors"] = actors
        faces_list.append(new_face_obj)

    response["faces"] = faces_list

    
    # ------- TMDB Algo -------

    # Convert from IMDB_ID to TMDB_ID
    tmdb_ids = convert_imdb_to_tmdb(imdb_ids, API_KEY)

    # Find the range of years of media we need to search through based on the algorithm we defined in Google Docs/Discord
    approximate_movie_dates = get_approximate_movie_dates(tmdb_ids, ages)
    approximate_year_range = calculate_release_range(approximate_movie_dates, error_margin=10)
    
    # Find the possible ranges of media using the tmdb_ids and year range

    possible_media = rank_media(approximate_year_range, tmdb_ids, 8)
    response["media"] = possible_media
    with open("output.txt", "w") as f:
        f.write(str(response))

    return response
