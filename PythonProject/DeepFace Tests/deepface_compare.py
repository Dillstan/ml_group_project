from deepface import DeepFace
import os

os.environ["DEEPFACE_POSTGRES_URI"] = "postgresql://deepface_user:password@localhost:5432/deepface_db"

res = DeepFace.search("nm0593961_underworld_1.jpeg", detector_backend='retinaface')[0]
res.to_csv("nm0593961_underworld_1.csv", index=False)

res = DeepFace.search("nm0593961_underworld_2.jpeg", detector_backend='retinaface')[0]
res.to_csv("nm0593961_underworld_2.csv", index=False)