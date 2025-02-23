import os
import cv2
import numpy as np

def read_images(path, sz=(300, 300)):
    """ Load training images and labels """
    X, y = [], []
    label = 0

    for filename in os.listdir(path):
        try:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(path, filename)
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

                img = cv2.resize(img, sz)
                X.append(np.asarray(img, dtype=np.uint8))
                y.append(label)

        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

    return [X, y]
