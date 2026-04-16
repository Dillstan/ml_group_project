from deepface import DeepFace

def extract_faces_from_image(image_path):
    """
    Imports the .jpg and detects/splits multiple faces.
    Using 'retinaface' or 'yolov8' is best for crowd/multi-face photos.
    """
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

def embed_face(face_img):
    """
    Takes a single cropped face and embeds it into 512D using ArcFace.
    """
    # DeepFace.represent returns a list of dictionaries; we grab the embedding vector
    embedding_obj = DeepFace.represent(
        img_path=face_img, 
        model_name="ArcFace",
        enforce_detection=False # Already detected in the extraction phase
    )
    return embedding_obj[0]['embedding']

def extract_faces_from_crowd(img_path):
    # DeepFace.represent returns a list of dictionaries; we grab the embedding vector
    embedding_obj = DeepFace.represent(
        img_path=img_path,
        model_name="ArcFace",
        enforce_detection=True ,
        detector_backend='retinaface'
    )
    return [e['embedding'] for e in embedding_obj]
