import cv2
import os

# Paths
INPUT_FOLDER = r"C:/Users/ZPHRRWA/Desktop/DEMO/dataset"  # Change if needed
OUTPUT_FOLDER = r"C:/Users/ZPHRRWA/Desktop/DEMO/cropped_faces"

face_cascades = [
    cv2.CascadeClassifier("C:/Users/ZPHRRWA/Desktop/DEMO/haarcascade_frontalface_default.xml"),
    cv2.CascadeClassifier("C:/Users/ZPHRRWA/Desktop/DEMO/haarcascade_frontalface_alt.xml"),
    cv2.CascadeClassifier("C:/Users/ZPHRRWA/Desktop/DEMO/haarcascade_frontalface_alt2.xml")
]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def detect_faces(gray_img):
    """Try detecting faces using both cascades"""
    for cascade in face_cascades:
        faces = cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)
        if len(faces) > 0:
            return faces
    return []  # No faces found

def crop_faces():
    """Detect and crop faces from images in INPUT_FOLDER and save to OUTPUT_FOLDER."""
    for subdir, _, files in os.walk(INPUT_FOLDER):
        for file in files:
            img_path = os.path.join(subdir, file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"⚠️ Skipping {file} (Invalid image)")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray)

            if len(faces) == 0:
                print(f"❌ No faces detected in {file}, skipping...")
                continue

            for i, (x, y, w, h) in enumerate(faces):
                face = img[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (200, 200))

                # Save cropped face
                save_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(file)[0]}_face{i}.jpg")
                cv2.imwrite(save_path, face_resized)
                print(f"✅ Saved: {save_path}")

    print("\n🎉 All detected faces have been cropped and saved!")

if __name__ == "__main__":
    crop_faces()
