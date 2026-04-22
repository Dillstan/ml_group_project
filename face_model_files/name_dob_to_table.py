import numpy as np
from scipy.io import loadmat
import psycopg2
from datetime import datetime

# --- CONFIGURATION ---
MAT_FILE_PATH = 'imdb.mat'
DB_CONFIG = {
    "dbname": "deepface_db",
    "user": "deepface_user",
    "password": "password",  # Ensure this is correct!
    "host": "localhost",
    "port": "5432"
}
TABLE_NAME = "arc_face.embeddings"


def matlab_to_python_date(matlab_datenum):
    """Converts MATLAB serial date to standard YYYY-MM-DD string."""
    try:
        if np.isnan(matlab_datenum) or matlab_datenum < 366:
            return None
        dt = datetime.fromordinal(int(matlab_datenum) - 366)
        return dt.strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return None


def enrich_database():
    print(f"Loading metadata from {MAT_FILE_PATH}...")
    mat_data = loadmat(MAT_FILE_PATH)
    imdb = mat_data['imdb'][0, 0]

    full_paths = imdb['full_path'][0]
    names = imdb['name'][0]
    dobs = imdb['dob'][0]

    # 1. Build the catalog
    print("Parsing MATLAB records...")
    actor_catalog = {}

    for i in range(len(full_paths)):
        raw_path = str(full_paths[i][0]).strip()
        filename = raw_path.split('/')[-1]
        actor_id = filename.split('_')[0]

        if actor_id not in actor_catalog:
            try:
                # Safe UTF-8 extraction
                if len(names[i]) > 0:
                    actor_name = str(names[i][0]).encode('utf-8', errors='ignore').decode('utf-8')
                else:
                    actor_name = None

                actor_dob = matlab_to_python_date(dobs[i])

                actor_catalog[actor_id] = {
                    "name": actor_name,
                    "dob": actor_dob
                }
            except Exception:
                pass

    total_actors = len(actor_catalog)
    print(f"Successfully compiled profiles for {total_actors} unique actors.")

    # 2. Connect and Loop
    print("Connecting to database via SSH Tunnel...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_client_encoding('UTF8')
        cursor = conn.cursor()

        print("Injecting data into PostgreSQL (One by one)...")

        count = 0
        for actor_id, info in actor_catalog.items():
            count += 1

            # Execute a single regular query
            cursor.execute(
                f"UPDATE {TABLE_NAME} SET actor_name = %s, dob = %s WHERE actor_id = %s;",
                (info["name"], info["dob"], actor_id)
            )

            # Print an update and save to the database every 500 rows
            if count % 500 == 0:
                conn.commit()
                print(f"   -> Saved {count} / {total_actors} profiles...")

        # Final commit for the remaining rows
        conn.commit()
        print("MISSION ACCOMPLISHED: Database successfully enriched!")

    except psycopg2.Error as e:
        print(f"\nCRITICAL DATABASE ERROR at row {count}: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


if __name__ == '__main__':
    enrich_database()