from face_model_files import deepface_functions as df

def execute_program_on_image(img_path, save_path):
    # ------- Models and Preprocessing -------

    #split faces and save to ../extracted_photos_temp
    faces = df.split_and_save(img_path, save_path)

    #Gets the IMDB_IDS and confidence values from the postgres_db
    imdb_ids = df.embed_and_search(faces)

    #GetAges will be in the age_model_files directory and will use the photos that were saved to extracted_photos_temp and predict their ages
    #ages = GetAges()


    # ------- TMDB Algo -------

    #Convert from IMDB_ID to TMDB_ID
    #tmdb_ids = GetTMDBIds(imdb_ids)

    #This step may be able to be condensed using the IMDB-WIKI dataset, so this is just heere as a placeholder for now...
    #use the TMDB_ID to get all info we can on the actor (dob, name are most important here)
    #actor_info = GetActorInfo(tmdb_ids)

    #Find the range of years of media we need to search through based on the algorithm we defined in Google Docs/Discord
    #year_range = GetRange(actor_dob, ages)

    #Find the possible ranges of media using the tmdb_ids and year range
    # Get Media will also be where Max's RankMedia() function is called.
    #possible_media = GetMedia(tmdb_ids, year_range)

    #this is what wer return to the UI for displaying
    #return possible_media, actor_info, imdb_ids