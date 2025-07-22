import cv2
import os
import numpy as np
import pandas as pd

IMG_SIZE = 224
DATA_PATH = 'dataset/images/'
CSV_PATH = 'dataset/train.csv'

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE()
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def load_and_preprocess_images():
    df = pd.read_csv(CSV_PATH)
    images, labels = [], []
    for _, row in df.iterrows():
        path = os.path.join(DATA_PATH, f"{row['id_code']}.png")
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = apply_clahe(img)
        images.append(img)
        labels.append(row['diagnosis'])
    return np.array(images), np.array(labels)