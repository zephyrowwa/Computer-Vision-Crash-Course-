import numpy as np
import os
import cv2

def read_images(path):
    X, y = [], []
    label = 0  # Start labeling from 0

    for person in sorted(os.listdir(path)):  # Sort to ensure consistent labeling
        person_path = os.path.join(path, person)
        if not os.path.isdir(person_path):
            continue  # Skip files, only process directories

        for filename in os.listdir(person_path):
            img_path = os.path.join(person_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

            if img is None:
                print(f"Could not read {img_path}, skipping.")
                continue

            img = cv2.resize(img, (200, 200))  # Resize for consistency
            X.append(np.asarray(img, dtype=np.uint8))
            y.append(label)

        label += 1  # Move to the next person

    return np.array(X), np.array(y)

if __name__ == "__main__":
    dataset_path = "C:/Users/ZPHRRWA/Desktop/7. Performing Face Recognition/dataset"  # Change if your dataset is in a different location
    X, y = read_images(dataset_path)

    print(f"Loaded {len(X)} images with {len(set(y))} unique labels.")