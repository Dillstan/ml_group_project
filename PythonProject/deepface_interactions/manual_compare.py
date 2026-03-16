import psycopg2 as pg
from deepface import DeepFace as df
import time

PG_URI = 'postgresql://deepface_user:password@localhost:5432/deepface_db'
IMG_LIST = ['./deepface_test_results/nm0001467.jpg',
            './deepface_test_results/nm0000195.jpg',
            './deepface_test_results/nm0593961.jpg',
            './deepface_test_results/nm0861361.jpg',
            './deepface_test_results/nm0593961_underworld_1.jpeg',
            './deepface_test_results/nm0593961_underworld_2.jpeg']

HIMYM_TEST = ['./deepface_test_results/himym1.jpeg',
              './deepface_test_results/himym2.jpeg',
              './deepface_test_results/himym3.jpeg',
              './deepface_test_results/himym4.jpeg',
              './deepface_test_results/himym5.jpeg',]

class cmp_result:
    def __init__(self, _id, _actor_id, _distance, _confidence, _orig_img_name):
        self.id = _id
        self.actor_id = _actor_id
        self.distance = _distance
        self.confidence = _confidence
        self.orig_img_name = _orig_img_name

    id = 0
    actor_id = ''
    distance = 0.0
    confidence = 0.0
    orig_img_name = ''

    def get_confidence_percent(self):
        return f"{self.confidence*100:.2f}%"
    def __str__(self):
        return f'--- --- ---\norig_img_name: {self.orig_img_name}\nactor_id: {self.actor_id}\ndistance: {self.distance}\nconfidence: {self.confidence}\n--- --- ---'

def embed_img(target_img):
    start_time = time.time()
    rep = df.represent(target_img, detector_backend='retinaface', model_name="ArcFace", enforce_detection=False)
    target = rep[0]['embedding']
    print(f"\n--- {(time.time() - start_time)} seconds to run extract embedding---", flush=True)
    return target

def quick_search(target_embedding, target_img_name):
    connection = pg.connect(PG_URI)
    cur = connection.cursor()

    query = '''
        select actor_id, embedding::vector <=> %s::vector distance from arc_face.embeddings
        order by distance asc
        limit 1
    '''

    start_time = time.time()
    cur.execute(query, (str(target_embedding),))
    rows = cur.fetchall()
    print(f"--- {(time.time() - start_time)} seconds to find match---", flush=True)

    results = []

    for row in rows:
        actor_id = row[0]
        distance = row[1]
        confidence = 1-distance
        results.append(cmp_result(id, actor_id, distance, confidence,target_img_name))

    return results


def main():
    for img in HIMYM_TEST:
        target = embed_img(img)
        res = quick_search(target,img)

        for r in res:
            print(str(r))

if __name__ == '__main__':
    main()


