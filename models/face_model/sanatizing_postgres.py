import numpy as np
from scipy.io import loadmat
import psycopg2

# --- CONFIGURATION ---
MAT_FILE_PATH = '../../../ml_project_files/imdb_crop/imdb.mat'
DB_CONFIG = {
    "dbname": "deepface_db",
    "user": "deepface_user",
    "password": "password",
    "host": "localhost"
}


def clean_database():
    print(f"Loading metadata from {MAT_FILE_PATH}...")

    mat_data = loadmat(MAT_FILE_PATH)
    imdb = mat_data['imdb'][0, 0]

    # Extract the arrays
    full_paths = [path[0] for path in imdb['full_path'][0]]
    second_face_scores = imdb['second_face_score'][0]

    # np.isnan() returns True if it's a single face (NaN). I want the ~ (NOT) NaN ones.
    print("Scanning for multiple faces...")
    polluted_indices = np.where(~np.isnan(second_face_scores))[0]

    # Gather the exact file paths of the polluted images
    polluted_paths = [full_paths[i].split('/')[-1] for i in polluted_indices]
    print(f"Found {len(polluted_paths)} polluted images containing multiple faces.")

    if len(polluted_paths) == 0:
        print("Dataset is already clean. Exiting.")
        return

    # Execute the Batch Deletion
    print("Connecting to PostgreSQL...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        delete_query = f"DELETE FROM arc_face.embeddings WHERE img_name = ANY(%s);"

        print("Executing mass deletion query (this may take a moment)...")
        cursor.execute(delete_query, (polluted_paths,))
        conn.commit()

        print(f"SUCCESS: Purged {cursor.rowcount} polluted vectors from the database!")

    except psycopg2.Error as e:
        print(f"Database error occurred: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


if __name__ == '__main__':
    clean_database()