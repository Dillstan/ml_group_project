import face_recognition
from PIL import Image, ImageOps
import os

#process one image
def process_single_image(file_name, dir_basename):
  try:
    print(f"starting process on {file_name}")
    image = face_recognition.load_image_file(os.path.join(f"./datasets/faces/{dir_basename}", file_name))
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
          (256,256),
          Image.Resampling.LANCZOS,
          (0,0,0) # black padding
      )

      processed_image.save(f"./datasets/faces/cropped/{file_name}")
      return (file_name, True, "face_found_and_saved")

    return (file_name, False, "no_face")

  except Exception as err:
    # Keep multiprocessing alive even when a single file is unreadable/corrupt.
    return (file_name, False, f"error:{type(err).__name__}")