import psycopg2 as pg
from deepface import DeepFace as df
import time
PG_URI = 'postgresql://deepface_user:password@localhost:5432/deepface_db'


class cmp_result:
    def __init__(self, _id, _actor_id, _distance, _confidence):
        self.id = _id
        self.actor_id = _actor_id
        self.distance = _distance
        self.confidence = _confidence

    id = 0
    actor_id = ''
    distance = 0.0
    confidence = 0.0

    def get_confidence_percent(self):
        return f"{self.confidence*100:.2f}%"
    def __str__(self):
        return f'--- --- ---\n\nactor_id: {self.actor_id}\ndistance: {self.distance}\nconfidence: {self.confidence}\n--- --- ---'

def quick_search(target_embedding,num_rows):
    connection = pg.connect(PG_URI)
    cur = connection.cursor()

    query = f'''
        with q1 as (
        select actor_id, embedding::vector <=> %s::vector distance from arc_face.embeddings
        )
        select * from q1
        where distance < 0.50
        order by distance asc
        limit {num_rows}
    '''

    cur.execute(query, (str(target_embedding),))
    rows = cur.fetchall()
    results = []

    for row in rows:
        actor_id = row[0]
        distance = row[1]
        confidence = 1-distance
        results.append(cmp_result(id, actor_id, distance, confidence))

    return results




