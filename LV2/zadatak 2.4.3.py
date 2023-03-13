import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("road.jpg")
img = img[:, :, 0].copy()
plt.figure()
plt.imshow(img, cmap="gray")
plt.title("original")
plt.show()

plt.figure()
plt.imshow(img, cmap="gray", alpha=0.5)
plt.title("Svijetlije")
plt.show()

a = img.shape[1]
img4 = img[:, int(a/4): int(a/2)]

plt.figure()
plt.imshow(img4, cmap="gray")
plt.title("Druga cetvrtina")
plt.show()

img90 = np.rot90(img)
img90 = np.rot90(img90)
img90 = np.rot90(img90)

plt.figure()
plt.imshow(img90, cmap="gray")
plt.title("90")
plt.show()

plt.figure()
plt.imshow(np.fliplr(img), cmap="gray")
plt.title("flip")
plt.show()
