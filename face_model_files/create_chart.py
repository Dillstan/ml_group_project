import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from deepface import DeepFace
from scipy.spatial.distance import cosine

# 1. Point this to the IMDB crop folder you already have on your Mac
BASE_DIR = '/Users/dilloncoker/Desktop/spring_26/machine_learning/ml_project_files/imdb_crop/01/'

print("Scanning local images...")
all_images = [f for f in os.listdir(BASE_DIR) if f.endswith('.jpg')]

# Generate 20 random pairs (You can increase this if you have time)
distances = []

print("Running 20 random comparisons...")
for i in range(20000):
    img1_path = os.path.join(BASE_DIR, random.choice(all_images))
    img2_path = os.path.join(BASE_DIR, random.choice(all_images))

    try:
        # Extract embeddings silently
        emb1 = DeepFace.represent(img1_path, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
        emb2 = DeepFace.represent(img2_path, model_name="ArcFace", enforce_detection=False)[0]["embedding"]

        # Calculate Cosine Distance
        dist = cosine(emb1, emb2)
        distances.append(dist)
    except Exception:
        continue  # Skip if no face is found

# 2. Plot the results!
print("Plotting chart...")
sns.kdeplot(distances, fill=True, color="blue")
plt.title("ArcFace Distance Distribution (Random IMDB Pairs)")
plt.xlabel("Cosine Distance (Higher = Less Alike)")
plt.ylabel("Density")
plt.axvline(x=0.5, color='red', linestyle='--', label="Threshold Limit")
plt.legend()
plt.show()  # This will pop up a window you can screenshot for your slide!