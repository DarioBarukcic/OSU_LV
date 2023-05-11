import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
#plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

#1.
print(np.unique(len(img_array_aprox)))

#2.
KM = KMeans(n_clusters=6, init="k-means++", n_init=5)
KM.fit(img_array_aprox)

#3.
labels = KM.predict(img_array_aprox)

newimg = KM.cluster_centers_[labels]
newimg = np.reshape(newimg, (img.shape))

plt.figure()
plt.title("Kvantizirana slika")
plt.imshow(newimg)
plt.tight_layout()
plt.show()

#4.
#manjim brojem centara smanjuje se kvaliteta slike

#5.
for i in range (2, 7):
    img = Image.imread(f"imgs\\test_{i}.jpg")        
    img = img.astype(np.float64) / 255
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))
    img_array_aprox = img_array.copy()
    km = KMeans(n_clusters=4, init='random', n_init=5, random_state=0)
    km.fit(img_array_aprox)
    labels = km.predict(img_array_aprox)
    newimg = km.cluster_centers_[labels]
    newimg = np.reshape(newimg, (img.shape))
    f, axarr = plt.subplots(2, 1)
    axarr[0].imshow(img)
    axarr[1].imshow(newimg)
    plt.show()

#6.

img = Image.imread("imgs\\test_1.jpg")

img = img.astype(np.float64) / 255

w, h, d = img.shape
img_array = np.reshape(img, (w*h, d))

img_array_aprox = img_array.copy()

Ks = range(2, 11)
J = []

for i in Ks:
    KM = KMeans(n_clusters=i, init="random", n_init=5)
    KM.fit(img_array_aprox)
    J.append(KM.inertia_)

plt.plot(Ks, J)
plt.show()

#7.
img = Image.imread("imgs\\test_1.jpg")

img = img.astype(np.float64) / 255

w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

img_array_aprox = img_array.copy()

km = KMeans(n_clusters=4, init='random', n_init=5, random_state=0)

km.fit(img_array_aprox)

labels = km.predict(img_array_aprox)

unique_labels = np.unique(labels)
print(unique_labels)

f, axarr = plt.subplots(2, 2)

for i in range(len(unique_labels)):
    bit_values = labels==[i]
    bit_img = np.reshape(bit_values, (img.shape[0:2]))
    bit_img = bit_img*1
    x=int(i/2)
    y=i%2
    axarr[x, y].imshow(bit_img)

plt.show()
