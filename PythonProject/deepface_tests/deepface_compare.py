from deepface import DeepFace
import os

os.environ["DEEPFACE_POSTGRES_URI"] = "postgresql://deepface_user:password@localhost:5432/deepface_db"

res = DeepFace.search("./deepface_test_results/nm0000195.jpg", detector_backend='retinaface')[0]
res.to_csv("./deepface_test_results/nm0000195.csv", index=False)
