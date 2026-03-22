import numpy as np

def check_liveness(prev_frame, current_frame, threshold=5.0):
    if prev_frame is None:
        return False

    diff = np.mean(np.abs(current_frame - prev_frame))

    if diff > threshold:
        return True
    return False