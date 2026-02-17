from deepface import DeepFace as df
import os
import itertools

# 1. Setup Database Connection
# Make sure your Postgres service is actually running! ('brew services start postgresql')
os.environ["DEEPFACE_POSTGRES_URI"] = "postgresql://deepface_user:password@localhost:5432/deepface_db"

print("Starting registration...")

with open("imgpaths", 'r') as f:
    x = 1800
    for line in itertools.islice(f, x, None):
        #normalize string by removing line ending
        img_path = line.strip()

        #if empty line, continue
        if not img_path: continue

        try:
            df.register(img_path, detector_backend='retinaface')
            x = x + 1
            if x % 100 == 0:
                print(f"{x} images added...")
        except Exception as e:
            print(f"FAILED on ({img_path}): {e}")