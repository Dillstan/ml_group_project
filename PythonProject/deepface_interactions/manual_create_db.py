from deepface import DeepFace as df
import os
import itertools
import datetime
import sys
import shutil
import psycopg2 as pg
import re

# initalize vars
LINE_START = 0
PATHS_FILE = './logs/db_reg_imgpaths_intersection'
PROGRESS_LOG_FILE = './logs/db_reg_progress_intersection'
ERROR_LOG_FILE = './logs/db_reg_error_intersection'
MIN_FREE_SPACE_MB = 1000
PG_URI = 'postgresql://deepface_user:password@localhost:5432/deepface_db'

insert_query = '''
    insert into arc_face.embeddings (actor_id, actor_dob_yr, actor_age, img_path, img_name, img_taken_yr, embedding)
    values (%s, %s, %s, %s, %s, %s, %s::vector);
'''


# --- HELPER FUNCTIONS ---
def check_disk_space():
    """Returns False if disk is full"""
    try:
        # Check the drive where the current script is running
        # OR put the path to your external drive here: "/Volumes/Samsung_T7"
        total, used, free = shutil.disk_usage("/Volumes/deepface_db")

        free_mb = free // (2 ** 20)
        if free_mb < MIN_FREE_SPACE_MB:
            print(f"CRITICAL: Low Disk Space! Only {free_mb}MB remaining.")
            return False
        return True
    except:
        return True  # Default to true if check fails (e.g. permission issue)

# Get all of the filepaths that already exist with the db
def get_existing_files_from_db():
    print("--- Syncing with Database... ---")
    try:
        # connect to PG db
        conn = pg.connect("postgresql://deepface_user:password@localhost:5432/deepface_db")
        cur = conn.cursor()

        # get file paths from table
        cur.execute("SELECT img_path FROM arc_face.embeddings")
        rows = cur.fetchall()

        #extract as list
        existing_set = {row[0] for row in rows}

        #close PG connectoin
        conn.close()
        print(f"--- Found {len(existing_set)} images already in DB. ---")
        return existing_set
    except Exception as e:
        print(f"--- DB Sync Failed: {e} ---")
        return set()

#Use regular expressions to extract important information from the image filepath
def get_img_info(img_path):
    match = re.search(r'(nm[0-9]+)', img_path)
    a_id = match.group(1) if match else None

    match = re.search(r'rm[0-9]+_([0-9]+)', img_path)
    a_dob_yr = match.group(1) if match else None

    match = re.search(r'.*_([0-9]+)', img_path)
    img_yr = match.group(1) if match else None

    match = re.search(r'.*/imdb_crop/[0-9]+/(.*)', img_path)
    name = match.group(1) if match else None

    return a_id, int(a_dob_yr), int(img_yr), name

# --- SCRIPT BODY ---
# If an argument is passed, use it. Otherwise, use default
if len(sys.argv) > 1:
    LINE_START = int(sys.argv[1])

#Setup PG connection and get existing files
done_files = get_existing_files_from_db()

#Open files
with open(PATHS_FILE, 'r') as f_paths:
    with open(PROGRESS_LOG_FILE, 'a') as f_prog:
        with open(ERROR_LOG_FILE, 'a') as f_err:

            # initalize vars
            images_added = LINE_START
            images_skipped = 0
            index = LINE_START

            #write init line
            init_message = f"\n\n--- starting registration at line {LINE_START} | {datetime.datetime.now()} ---\n"
            f_prog.write(init_message)
            f_err.write(init_message)
            print(init_message)

            #ensure save even in event of crash
            f_prog.flush()
            f_err.flush()

            # connect to PG
            conn = pg.connect(PG_URI)
            cur = conn.cursor()

            #loop starting at specified line using islice function from itertools.
            for line in itertools.islice(f_paths, LINE_START, None):
                index = index + 1

                #remove str line ending
                img_path = line.strip()

                #if empty line, continue
                if not img_path: continue

                if img_path in done_files:
                    images_added += 1
                    images_skipped += 1
                    continue

                try:
                    #embed image using retinaface, ArcFace, and enforced detection of faces
                    embedding_res = df.represent(
                        img_path,
                        detector_backend='retinaface',
                        model_name='ArcFace',
                        enforce_detection=True)

                    #extrace the 512D embedding from result
                    embedding_512 = embedding_res[0]['embedding']

                    # get photo data
                    actor_id, actor_dob_yr, img_taken_yr, img_name = get_img_info(img_path)

                    #insert into db
                    pg_res = cur.execute(
                        insert_query,
                        (actor_id, actor_dob_yr, img_taken_yr - actor_dob_yr, img_path, img_name, img_taken_yr, str(embedding_512))
                    )

                    #increase images added
                    images_added += 1

                    # if multiple of 100 print and write to file
                    if images_added % 100 == 0:
                        conn.commit()
                        prog_message = f'{images_added} - {datetime.datetime.now()}\n'
                        f_prog.write(prog_message)
                        f_prog.flush()
                        print(prog_message)

                        if images_skipped > 0:
                            print(f'{images_skipped} images skipped due to overlap during cycle...\n\n')
                            images_skipped = 0

                        if not check_disk_space():
                            sys.exit(77)  # Custom Exit Code for "Disk Full"

                # on exception write to file
                except Exception as e:
                    conn.commit()
                    err_message = f"FAILED | {datetime.datetime.now()} | Index: {index}, File: {img_path} | {e}\n"
                    f_err.write(err_message)
                    f_err.flush()
                    print(err_message)

                #ensure all records are commited to db
                conn.commit()