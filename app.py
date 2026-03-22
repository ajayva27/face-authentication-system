import streamlit as st
import cv2
import numpy as np
import pickle
import os
import time

from keras_facenet import FaceNet
from mtcnn import MTCNN

from services.logging_service import log_event
from utils.face_detection import extract_faces
from utils.embedding import get_embedding
from utils.verification import identify_face, verify_face
from utils.faiss_utils import build_faiss_index, faiss_search


# LOAD MODELS 
@st.cache_resource
def load_models():
    embedder = FaceNet()
    detector = MTCNN()
    return embedder, detector

embedder, detector = load_models()


# LOAD DATABASE 
DB_PATH = "database.pkl"

def load_database():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    return {}

def save_database(db):
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)

database = load_database()
faiss_index, faiss_labels = build_faiss_index(database)


# UI CONFIG 
st.set_page_config(page_title="Face Authentication System", layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 900px;
}
</style>
""", unsafe_allow_html=True)

st.title("🔐 Face Authentication System")
st.markdown("Smart face recognition and verification system")


# SIDEBAR 
st.sidebar.markdown("## 🧭 Navigation")

option = st.sidebar.radio(
    "",
    ["📸 Recognition", "➕ Add Person", "🎥 Live Verification", "📊 Logs"]
)



# 1. FACE RECOGNITION
if option == "📸 Recognition":
    st.header("📸 Face Recognition")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        faces, boxes = extract_faces(img, detector)

        detected_names = []
        distances = []

        for face, (x, y, w, h) in zip(faces, boxes):
            emb = get_embedding(face, embedder)
            # name, dist = identify_face(emb, database)
            name, dist = faiss_search(emb, faiss_index, faiss_labels)

            detected_names.append(name)
            distances.append(dist)

            # Color based on result
            if name == "Unknown":
                color = (0, 0, 255)
                label = "Unknown"
            else:
                color = (0, 255, 0)
                label = f"{name} ({dist:.2f})"

            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

            cv2.putText(
                img,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA
            )

        # Unique names (correct)
        unique_names = list(set(detected_names))

        # LOG ONCE (clean)
        if unique_names:
            for i, name in enumerate(unique_names):
                if name != "Unknown":
                    log_event(name, "SUCCESS", "Recognition", distances[i])
                else:
                    log_event("Unknown", "FAILED", "Recognition", distances[i])

        # Display
        if unique_names:
            if unique_names == ["Unknown"]:
                st.error("No known person detected")
            else:
                st.success(f"Detected: {', '.join(unique_names)}")
        else:
            st.warning(" No face detected")

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# 🔹 2. ADD PERSON

elif option == "➕ Add Person":
    st.header("➕ Add New Person")

    name = st.text_input("Enter Name")

    uploaded_files = st.file_uploader(
        "Upload Images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    if st.button("Add to Database"):
        if name == "" or not uploaded_files:
            st.warning(" Provide name and images")
        else:
            if name not in database:
                database[name] = []

            count = 0

            for file in uploaded_files:
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)

                faces, _ = extract_faces(img, detector)

                if len(faces) > 0:
                    emb = get_embedding(faces[0], embedder)
                    database[name].append(emb)
                    count += 1

            # optimize embeddings
            if len(database[name]) > 1:
                database[name] = [np.mean(database[name], axis=0)]

            save_database(database)

            faiss_index, faiss_labels = build_faiss_index(database)

            st.success(f" {count} images added for {name}")



# 🔹 3. LIVE VERIFICATION

elif option == "🎥 Live Verification":
    st.header("🎥 Live Face Verification")

    if len(database.keys()) == 0:
        st.warning(" No users in database")
        st.stop()

    person_name = st.selectbox("Select Person", list(database.keys()))

    st.markdown("### 🧍 Position your face inside the circle")

    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])
    status_text = st.empty()

    cap = cv2.VideoCapture(0)

    verified = False
    logged = False
    start_time = None

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera error")
            break

        h, w, _ = frame.shape

        center = (w//2, h//2)
        radius = 150

        circle_color = (0, 0, 255)  # red default

        faces, boxes = extract_faces(frame, detector)

        for face, (x, y, w_box, h_box) in zip(faces, boxes):
            emb = get_embedding(face, embedder)

            if start_time is None:
                start_time = time.time()

            elapsed = time.time() - start_time

            if elapsed > 1.5:
                # is_verified, dist = verify_face(emb, person_name, database)
                name, dist = faiss_search(emb, faiss_index, faiss_labels)

                is_verified = (name == person_name)

                if is_verified:
                    circle_color = (0, 255, 0)  # 🟢 green

                    status_text.success(f" Verified: {person_name}")

                    if not logged:
                        log_event(person_name, "SUCCESS", "Verification", dist)
                        logged = True

                    verified = True

                else:
                    status_text.error("Not Verified")

                    if not logged:
                        log_event(person_name, "FAILED", "Verification", dist)
                        logged = True

        cv2.circle(frame, center, radius, circle_color, 3)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if verified:
            break

    cap.release()


# 🔹 4. LOGS

elif option == "📊 Logs":
    st.header("📊 Attendance Logs")

    import pandas as pd

    log_path = "logs/attendance.csv"

    # Case 1: file doesn't exist
    if not os.path.exists(log_path):
        st.warning("No logs available yet")
    
    # Case 2: file exists but empty
    elif os.stat(log_path).st_size == 0:
        st.warning("Log file is empty")

    else:
        try:
            df = pd.read_csv(log_path)

            # Case 3: file has data
            st.dataframe(df, use_container_width=True)

            # Optional summary
            st.markdown("### 📈 Summary")

            col1, col2 = st.columns(2)
            col1.metric("Total Records", len(df))
            col2.metric("Successful", (df["Status"] == "SUCCESS").sum())

        except pd.errors.EmptyDataError:
            st.warning("Log file has no valid data")