import numpy as np
from datetime import datetime
from utils.verification import verify_face
from services.logging_service import log_event

def authenticate(face_embedding, person_name, database):
    is_verified, dist = verify_face(face_embedding, person_name, database)

    if is_verified:
        log_event(person_name, "SUCCESS")
        return True, dist
    else:
        log_event(person_name, "FAILED")
        return False, dist