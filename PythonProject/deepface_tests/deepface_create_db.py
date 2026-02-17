from deepface import DeepFace as df
import os
import itertools

# initalize vars
LINE_START = 8600

# 1. Setup Database Connection
# Make sure your Postgres service is actually running! ('brew services start postgresql')
os.environ["DEEPFACE_POSTGRES_URI"] = "postgresql://deepface_user:password@localhost:5432/deepface_db"

print("Starting registration...")

with open("./logs/db_reg_imgpaths", 'r') as f:
    with open("./logs/db_reg_progress", 'a') as ff:
        with open("./logs/db_reg_error", 'a') as ff2:
            # initalize vars
            images_added = LINE_START
            index = LINE_START

            #write init line
            init_message = f"\n\n--- starting registration at line {LINE_START} ---\n"
            ff.write(init_message)
            ff2.write(init_message)
            print(init_message)

            #loop starting at specified line
            for line in itertools.islice(f, LINE_START, None):
                index = index + 1

                #remove str line ending
                img_path = line.strip()
                #if empty line, continue
                if not img_path: continue

                try:
                    #register to db
                    df.register(img_path, detector_backend='retinaface')

                    #increase images added
                    images_added = images_added + 1

                    # if multiple of 100 print and write to file
                    if images_added % 100 == 0:
                        ff.write(str(images_added))
                        print(f"--- {images_added} images added ---\n")

                # on exception write to file
                except Exception as e:
                    ff2.write(f"FAILED | Index: {index}, File: {img_path} | {e}\n")