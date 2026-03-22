import cv2

def extract_faces(image, detector):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)

    faces = []
    boxes = []

    for res in results:
        x, y, w, h = res['box']
        x, y = abs(x), abs(y)

        face = img_rgb[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))

        faces.append(face)
        boxes.append((x, y, w, h))

    return faces, boxes