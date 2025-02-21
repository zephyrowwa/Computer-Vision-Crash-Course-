import sys
import numpy as np
import cv2
from readimg import read_images  # Import function from the first script

def face_rec():
    names = ['Lana Del Rey', 'Ryan Gosling']  # Change these to actual names

    dataset_path = "C:/Users/ZPHRRWA/Desktop/7. Performing Face Recognition/dataset"  # Change if needed
    X, y = read_images(dataset_path)
    y = np.asarray(y, dtype=np.int32)

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(X, y)

    camera = cv2.VideoCapture(0)  # Open webcam
    face_cascade = cv2.CascadeClassifier('C:/Users/ZPHRRWA/Desktop/7. Performing Face Recognition/haarcascade_frontalface_default.xml')

    while True:
        ret, img = camera.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (200, 200))

            try:
                label, confidence = model.predict(face_resized)
                label_name = names[label] if label < len(names) else "Unknown"
                cv2.putText(img, f"{label_name}, {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            except:
                continue

        cv2.imshow("Face Recognition", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_rec()
