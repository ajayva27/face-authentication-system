import faiss
import numpy as np
from config import FAISS_THRESHOLD


def build_faiss_index(database):
    embeddings = []
    labels = []

    for name, embs in database.items():
        for emb in embs:
            embeddings.append(emb)
            labels.append(name)

    if len(embeddings) == 0:
        return None, []

    embeddings = np.array(embeddings).astype("float32")

    # 🔥 IMPORTANT: Normalize embeddings
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, labels


def faiss_search(face_embedding, index, labels, threshold=FAISS_THRESHOLD):
    if index is None or len(labels) == 0:
        return "Unknown", None

    query = np.array([face_embedding]).astype("float32")

    # Normalize query too
    faiss.normalize_L2(query)

    distances, indices = index.search(query, k=1)

    dist = distances[0][0]
    idx = indices[0][0]

    # Safety check
    if idx < 0 or idx >= len(labels):
        return "Unknown", dist

    name = labels[idx]

    if dist > threshold:
        return "Unknown", dist

    return name, dist