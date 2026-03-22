import csv
from datetime import datetime
import os

LOG_PATH = "logs/attendance.csv"

os.makedirs("logs", exist_ok=True)

def log_event(name, status, mode, distance):
    file_exists = os.path.isfile(LOG_PATH)

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        # Write header ONLY if file doesn't exist
        if not file_exists:
            writer.writerow(["Name", "Time", "Status", "Mode", "Distance"])

        writer.writerow([
            name,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            status,
            mode,
            round(distance, 3) if distance is not None else None
        ])