import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Silence the warning spam too

from tmdb_files import tmdb_algorithm as tmdb

<<<<<<< HEAD
IMG_PATH = './~input_photos_temp/internship1.jpg'
=======
IMG_PATH = './~input_photos_temp/sienfled.jpg'
>>>>>>> db9176c (algorithm completion)
SAVE_PATH = './~extracted_photos_temp'

if __name__ == '__main__':
    tmdb.execute_program_on_image(img_path=IMG_PATH, save_path=SAVE_PATH)
    for filename in os.listdir(SAVE_PATH): # remove files after running model
        file_path = os.path.join(SAVE_PATH, filename)

        if os.path.isfile(file_path):
            os.remove(file_path)