from tmdb_files import tmdb_algorithm as tmdb

IMG_PATH = './~input_photos_temp/multiface_2.jpg'
SAVE_PATH = './~extracted_photos_temp'

if __name__ == '__main__':
    tmdb.execute_program_on_image(img_path=IMG_PATH, save_path=SAVE_PATH)
    #delete temp photos