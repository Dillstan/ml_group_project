from deepface import DeepFace
import concurrent.futures
import time
import numpy as np

# --- 1. IMPORT & SPLIT FACES ---
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

# --- 2. EMBED A SINGLE FACE ---
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

# --- 3. MOCK DATABASE QUERY ---
def query_database(embedding, threshold=0.3):
    """
    MOCK FUNCTION: This is where you will eventually run your pgvector SQL query.
    For now, we simulate a delay to test the multithreading.
    """
    # TODO: Connect to Dillon's pgvector DB and do a cosine distance search
    # e.g., SELECT name FROM actors ORDER BY embedding <=> '[...]' LIMIT 1;
    time.sleep(0.5) # Simulating database query time
    
    return {"status": "success", "match": "Simulated Actor Name", "confidence": 0.95}

# --- 4. THE MULTI-THREADED WORKER ---
def process_single_face(face_img, face_id):
    """
    The complete pipeline for one face: Embed -> Query
    """
    print(f"[Thread {face_id}] Starting embedding...")
    vector_512d = embed_face(face_img)
    
    print(f"[Thread {face_id}] Embedding complete. Querying database...")
    result = query_database(vector_512d)
    
    print(f"[Thread {face_id}] Query complete: {result['match']}")
    return {"face_id": face_id, "result": result}

# --- 5. THE MAIN SEARCH FUNCTION ---
def search_image(image_path):
    """
    Executes the entire facial extraction and search pipeline on a single image.
    Wrapped in a function so it can be imported by video_pipeline.py.
    """
    # Step A: Extract all faces (Sequential)
    faces = extract_faces_from_image(image_path)
    
    if faces:
        print("\n--- Starting Multi-threaded Embedding & Querying ---")
        start_time = time.time()
        
        # Step B: Multithreaded Embedding and Querying
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(faces)) as executor:
            # Submit all faces to the thread pool
            futures = [executor.submit(process_single_face, face, i) for i, face in enumerate(faces)]
            
            # Gather results as they finish
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                
        end_time = time.time()
        print(f"Total Processing Time for {len(faces)} faces: {end_time - start_time:.2f} seconds")
        return results
    
    return []

# --- MAIN EXECUTION (Local Testing Only) ---
if __name__ == "__main__":
    # It keeps our old single-image test working if we ever just run this file by itself.
    TEST_IMAGE = "../Test Images/Creed3.jpg"
    search_image(TEST_IMAGE)