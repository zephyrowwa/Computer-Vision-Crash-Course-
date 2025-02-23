import numpy as np
import cv2
import os
from readimg_demo import read_images 

DATASET_PATH = r"C:/Users/ZPHRRWA/Desktop/DEMO/dataset"

# Load both frontal face cascades
face_cascades = [
    cv2.CascadeClassifier('C:/Users/ZPHRRWA/Desktop/DEMO/haarcascade_frontalface_default.xml'),
    cv2.CascadeClassifier('C:/Users/ZPHRRWA/Desktop/DEMO/haarcascade_frontalface_alt.xml'),
    cv2.CascadeClassifier('C:/Users/ZPHRRWA/Desktop/DEMO/haarcascade_frontalface_alt2.xml')
]

def detect_faces(gray_img):
    """ Try detecting faces using both cascades """
    for cascade in face_cascades:
        faces = cascade.detectMultiScale(gray_img, scaleFactor=1.5, minNeighbors=5,
                                         minSize=(50,50), maxSize=(250,250)) # adjust mo den to 
        if len(faces) > 0:
            return faces
    return []

def face_rec():
    """ Face recognition script """
    names = ["ME"]

    # Load images using readimg_demo.py
    X, y = read_images(DATASET_PATH)
    y = np.asarray(y, dtype=np.int32)

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(X, y)

    camera = cv2.VideoCapture(0)  # Start webcam

    while True:
        ret, img = camera.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)

        for (x, y, w, h) in faces:
            roi = cv2.resize(gray[y:y + h, x:x + w], (300, 300), interpolation=cv2.INTER_LINEAR)

            try:
                label, confidence = model.predict(roi)

                if confidence < 70:  # Adjust threshold for better recognition
                    color = (255, 0, 0)  # Blue for recognized faces
                    text = names[label] + f" ({confidence:.2f})"
                else:
                    color = (0, 0, 255)  # Red for unknown faces
                    text = "Who is that?"

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, text, (x, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            except:
                continue

        cv2.imshow("Unggabungga", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_rec()
