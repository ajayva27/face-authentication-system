from sklearn.metrics.pairwise import cosine_similarity

def cosine_distance(e1, e2):
    return 1 - cosine_similarity([e1], [e2])[0][0]


def identify_face(face_embedding, database, threshold=0.4):
    min_dist = float("inf")
    identity = "Unknown"

    for name, embeddings in database.items():
        for db_emb in embeddings:
            dist = cosine_distance(face_embedding, db_emb)

            if dist < min_dist:
                min_dist = dist
                identity = name

    if min_dist > threshold:
        identity = "Unknown"

    return identity, min_dist


def verify_face(face_embedding, person_name, database, threshold=0.4):
    if person_name not in database:
        return False, None

    distances = []

    for db_emb in database[person_name]:
        dist = cosine_distance(face_embedding, db_emb)
        distances.append(dist)

    min_dist = min(distances)

    if min_dist < threshold:
        return True, min_dist
    else:
        return False, min_dist