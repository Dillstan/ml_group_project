import cv2
from deepface import DeepFace
import numpy as np
from face_model_files import postgres_functions as psql
import os

# Pads a rectangular image to a square, then resizes safely.
def pad_to_square(img, target_size=224):
    h, w = img.shape[:2]
    
    # 1. Find the longest side
    max_dim = max(h, w)
    
    # 2. Calculate how much padding is needed to make it a perfect square
    top = (max_dim - h) // 2
    bottom = max_dim - h - top
    left = (max_dim - w) // 2
    right = max_dim - w - left
    
    # 3. Add black padding (value=[0, 0, 0]) to the shorter sides
    squared_img = cv2.copyMakeBorder(
        img, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    
    # 4. Now that it is a perfect square, resizing it won't stretch the face
    return cv2.resize(squared_img, (target_size, target_size))

#Get faces from images returns the individual face images in an array
def extract_faces_from_image(image_path):
    print(f"Extracting faces from {image_path}...")
    try:
        # DeepFace extracts faces and returns a list of face objects
        face_objs = DeepFace.extract_faces(
            img_path=image_path, 
            detector_backend='retinaface', # Excellent at finding multiple faces
            enforce_detection=True
        )
        # We just need the face image data to pass to the embedder
        extracted_faces = [face['face'] for face in face_objs]
        print(f"Found {len(extracted_faces)} faces.")
        return extracted_faces
    except ValueError:
        print("No faces found in the image.")
        return []

#embed a single cropped face into a vector
def embed_face(face_img):
    # DeepFace.represent returns a list of dictionaries; we grab the embedding vector
    embedding_obj = DeepFace.represent(
        img_path=face_img, 
        model_name="ArcFace",
        enforce_detection=False # Already detected in the extraction phase
    )
    return embedding_obj[0]['embedding']

#Extact and embed all faces from a file as array of vectors
def extract_faces_from_crowd(img_path):
    # DeepFace.represent returns a list of dictionaries; we grab the embedding vector
    embedding_obj = DeepFace.represent(
        img_path=img_path,
        model_name="ArcFace",
        enforce_detection=True ,
        detector_backend='retinaface'
    )
    return [e['embedding'] for e in embedding_obj]

def split_and_save(img_path,save_path):
    print("Splitting and saving faces...")
    img_list = extract_faces_from_image(img_path)
    res = []
    for face_id, img in enumerate(img_list):
        img = pad_to_square(img, 224)
        # reverse the normalization and cast back to standard 8-bit image data
        face_img_ready = (img * 255).astype(np.uint8)

        # to be perfectly safe with OpenCV color spaces, ensure it is BGR
        # DeepFace extraction might return RGB depending on the version
        if face_img_ready.shape[-1] == 3:
            face_img_ready = face_img_ready[:, :, ::-1]

        #save rgb image to temp folder for Age Model to use
        cv2.imwrite(save_path+f'/{face_id}.jpg', face_img_ready)

        # return the results of deepface_split as array
    return img_list

#here we take an array of images and embed them individually, this takes milliseconds after the faces are extracted.
#list_of_faces requires the array returned from split_and_save
def embed_and_search(list_of_faces, top_n_faces=1, distance_threshold=0.5):
    res = []
    print("Embedding faces...")
    for face in list_of_faces:
        target = embed_face(face)
        res.append(psql.quick_search(target, top_n_faces, distance_threshold))
    return res