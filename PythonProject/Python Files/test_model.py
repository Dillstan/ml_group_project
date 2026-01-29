#SOME GPT CODE, MOSTLY MIND

from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from scipy.io import loadmat  # Import this at the beginning of your script
from collections import namedtuple

MODEL_PATH = "../../../ml_project_files/Models/mini_test_2/epoch/my_model_epoch_02.keras"
# Specify the image size
IMG_SIZE = (160, 160)
# Load the model from the .h5 file
loaded_model = load_model(MODEL_PATH)
Datapoint = namedtuple('Datapoint', 'id name year')

def load_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)
    return img

def predict(img, datapoint_map:Datapoint):
    # Make predictions
    predictions = loaded_model.predict(img)

    # Get the predicted class IDs
    predicted_class_ids = np.argmax(predictions, axis=1)

    # Retrieve and print the predicted name
    for predicted_class_id in predicted_class_ids:
        tpl = datapoint_map.get(predicted_class_id, "Unknown")  # Default to "Unknown" if ID not found
        print(f"Predicted class_id (for the ML): {predicted_class_id}:\n\tPredicted IMDB ID: {tpl.id}\n\tPredicted Name: {tpl.name},\n\tPredicted DOB: {tpl.year}\n")


# Load the `.mat` file directly
print("Loading imdb.mat...")
data = loadmat('../../../ml_project_files/imdb_crop/imdb.mat')  # Use your actual path to the file
imdb = data['imdb']



# Extract celebrity names from the loaded datafile_paths_raw = file_paths_raw.item()[0]
celeb_names = np.squeeze(imdb['name']).item()[0]  # Adjust if necessary
celeb_ids = np.squeeze(imdb['celeb_id']).item()[0]
dob_offsets = np.squeeze(imdb['dob']).item()[0]

print(len(celeb_names), len(dob_offsets), len(celeb_ids))

import datetime

# Example serial date number (days since January 1, 1970)
def matlab_to_datetime(matlab_date):
    return datetime.datetime(1, 1, 1) + datetime.timedelta(days=int(matlab_date-1))


# Use the celeb_names to create the mapping
datapoint_map = {i: Datapoint(celeb_ids[i], celeb_names[i], matlab_to_datetime(dob_offsets[i])) for i in range(len(celeb_names))}

# Load and preprocess the image
img1 = load_and_preprocess("../Test Images/OIP-2155502055.jpg")
img2 = load_and_preprocess("../Test Images/img.jpg")
img3 = load_and_preprocess("../Test Images/cilian.jpg")

# Make predictions
predict(img1, datapoint_map)
predict(img2, datapoint_map)
predict(img3, datapoint_map)
