import face_recognition
from PIL import Image, ImageOps
import os

#process one image
def process_single_image(file_name):
  image = face_recognition.load_image_file(os.path.join("./datasets/faces/images", file_name))
  face_locations = face_recognition.face_locations(image, model="hog")
  if len(face_locations) > 0: # face found in image

    # Get Cropped Image
    top, right, bottom, left = face_locations[0]
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)

    # add padding
    processed_image = ImageOps.pad(
        pil_image,
        (160,160),
        Image.Resampling.LANCZOS,
        (0,0,0) # black padding
    )

    processed_image.save(f"./datasets/faces/cropped/{file_name}")
    return (file_name, True, "face_found_and_saved")

  else: # image not successfully cropped
    print(file_name)
    return (file_name, False, "no_face")