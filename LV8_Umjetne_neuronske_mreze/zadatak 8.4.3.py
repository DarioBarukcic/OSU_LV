import numpy as np
from tensorflow import keras
from keras import layers         
from matplotlib import pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
import matplotlib.image as Image

model = load_model("FCN/")
model.summary()

img = Image.imread("Slika.png")
img = img[:, :, 0]

img_reshaped = np.reshape(img, (1, img.shape[0]*img.shape[1]))

img_pred = model.predict(img_reshaped)
print(img_pred)
img_pred = np.argmax(img_pred, axis=1)

print("Broj na slici:", img_pred)