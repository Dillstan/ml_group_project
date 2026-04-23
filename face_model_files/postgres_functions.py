import psycopg2 as pg
DB_CONFIG = {
    "dbname": "deepface_db",
    "user": "deepface_user",
    "password": "password",
    "host": "localhost",
    "port": "5432"
}


class cmp_result:
    def __init__(self, _id, _actor_id, _distance, _confidence, name, dob):
        self.id = _id
        self.actor_id = _actor_id
        self.distance = _distance
        self.confidence = _confidence
        self.dob

    id = 0
    actor_id = ''
    distance = 0.0
    confidence = 0.0
    name = ""

    def get_confidence_percent(self):
        return f"{self.confidence*100:.2f}%"
    def __str__(self):
        return f'--- --- ---\n\nactor_id: {self.actor_id}\ndistance: {self.distance}\nconfidence: {self.confidence}\n--- --- ---'

def quick_search(target_embedding, top_n_faces, distance_threshold):
    connection = pg.connect(**DB_CONFIG)
    cur = connection.cursor()

    query = f'''
        with q1 as (
        select actor_id, embedding::vector <=> %s::vector distance, name, dob from arc_face.embeddings
        )
        select * from q1
        where distance < {distance_threshold}
        order by distance asc
        limit {top_n_faces}
    '''

    cur.execute(query, (str(target_embedding),))
    rows = cur.fetchall()
    results = []

    for row in rows:
        actor_id = row[0]
        distance = row[1]
        confidence = 1-distance
        name = row[2]
        dob = row[3]
        results.append(cmp_result(id, actor_id, distance, confidence, name, dob))

    return results




