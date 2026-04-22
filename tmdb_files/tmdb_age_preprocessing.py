# Age Pre-processing for TMDB algorithm - Maxwell Klema
# April 21st, 2026

import face_recognition
from PIL import Image, ImageOps
import os
from pathlib import Path
import concurrent.futures
from itertools import repeat
from tensorflow import keras
import numpy as np
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import register_keras_serializable
from natsort import os_sorted

# Custom metric to track percent of age predictions correct within N-year tolerance.
@register_keras_serializable(package="Custom")
class WithinNYears(tf.keras.metrics.Metric):
    def __init__(self, tolerance=None, dtype=None, name=None, **kwargs):
        # Backward-compatible load path: if older saved configs omit tolerance
        if tolerance is None and isinstance(name, str):
            match = re.match(r"within_(\d+)_years", name)
            if match:
                tolerance = int(match.group(1))

        if tolerance is None:
            tolerance = 2

        super().__init__(name=name or f"within_{tolerance}_years", dtype=dtype, **kwargs)
        self.tolerance = int(tolerance)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.squeeze(y_pred, axis=-1) # get last tensor (output) and transform (batch_size,1) to (batch_size,)
        y_true = tf.cast(y_true, tf.float32) 
        within = tf.abs(y_true - y_pred) <= self.tolerance
        self.correct.assign_add(tf.reduce_sum(tf.cast(within, tf.float32))) # convert booleans to floats and get total sum and add to correct
        self.total.assign_add(tf.cast(tf.size(y_true), tf.float32)) # total is just size of y_true

    def result(self):
        return self.correct / self.total
    
    def reset(self): # runs once per epoch
        self.correct.assign(0)
        self.total.assign(0)

    def get_config(self):
        config = super().get_config()
        config.update({"tolerance": self.tolerance})
        return config
    
# optional method to pre-process images for age_model directly
def process_single_image(file_name, dir_basename):
    try:
        image = face_recognition.load_image_file(os.path.join(dir_basename, file_name))
        face_locations = face_recognition.face_locations(image, model="hog")
        if len(face_locations) > 0: # face found in image

            # Get Cropped Image
            height, width = image.shape[:2]
            top, right, bottom, left = face_locations[0]

            box_h = bottom - top
            box_w = right - left
            pad_h = int(box_h * 0.33) # add padding to get hair, neck, etc.
            pad_w = int(box_w * 0.33)

            top = max(0, top-pad_h)
            bottom = min(height, bottom+pad_h)
            left = max(0, left-pad_w)
            right = min(width, right+pad_w)

            face_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)

            # add padding
            processed_image = ImageOps.pad(
                pil_image,
                (224,224),
                Image.Resampling.LANCZOS,
                (0,0,0) # black padding
            )

            os.makedirs(f"./~temp_age_preprocessed", exist_ok=True)
            processed_image.save(f"./~temp_age_preprocessed/{file_name}")
            return (file_name, True)
        
        return (file_name, False) # no face detected

    except Exception as err:
        print(err)
        return (file_name, False)

def get_ages(image_dir, max_workers=1):
    images = os.listdir(image_dir)
    images = os_sorted(images) # case-insensitive sorting
    predicted_ages = []

    # can optionally crop and pre-process face images itself
    # however, image_dir should already contain 224 x 224 cropped images

    # if max_workers == 1:    
    #     for img in images:
    #         process_single_image(img, image_dir)

    # with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     executor.map(process_single_image, images, repeat(image_dir))

    # load age model

    model_path = "./models/age_models/cnn_model_best.keras"
    age_model = keras.models.load_model(model_path)

    for image in images:
        img = Image.open(os.path.join(image_dir, image))
        img_array = np.array(img,dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)

        pred = age_model.predict(img_batch) # call model
        print(f"{pred} - {image}")
        predicted_ages.append(round(pred[0][0]))

    return predicted_ages


print(get_ages("./~temp_age_preprocessed", max_workers=10))



