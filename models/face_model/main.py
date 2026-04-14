import deepface_functions as df
import postgres_functions as psql
import cv2
import matplotlib.pyplot as plt
import numpy as np

IMG_LIST = ['./Test Images/nm0001467.jpg',
            './Test Images/nm0000195.jpg',
            './Test Images/nm0593961.jpg',
            './Test Images/nm0861361.jpg',
            './Test Images/nm0593961_underworld_1.jpeg',
            './Test Images/nm0593961_underworld_2.jpeg']

HIMYM_TEST = ['./Test Images/himym1.jpeg',
              './Test Images/himym2.jpeg',
              './Test Images/himym3.jpeg',
              './Test Images/himym4.jpeg',
              './Test Images/himym5.jpeg',]

IVIE_TEST = ['./Test Images/nm0728812.jpg']

def main():
    img_list = df.extract_faces_from_image('./Test Images/mister_doubting_flames.jpg')
#    img_list = HIMYM_TEST
#    img_list = df.extract_faces_from_crowd('./Test Images/multiface_2.jpg')
    for face_id, img in enumerate(img_list):
        # 1. Reverse the normalization and cast back to standard 8-bit image data
        face_img_ready = (img * 255).astype(np.uint8)

        # 2. To be perfectly safe with OpenCV color spaces, ensure it is BGR
        # DeepFace extraction sometimes returns RGB depending on the version
        if face_img_ready.shape[-1] == 3:
            face_img_ready = face_img_ready[:, :, ::-1]

        target = df.embed_face(face_img_ready)
        res = psql.quick_search(target, 1)

        print("Top 5 Results for face_id:" + str(face_id))
        for r in res:
            print(str(r))
        print()
        print()

if __name__ == '__main__':
    main()
