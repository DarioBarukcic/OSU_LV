import numpy as np
import matplotlib.pyplot as plt

bijelo = np.zeros((50, 50))
crno = np.ones((50, 50))


gornjePolje = np.hstack((crno, bijelo))
donjePolje = np.hstack((bijelo, crno))

polje = np.vstack((gornjePolje, donjePolje))
print(polje)

plt.figure()
plt.imshow(polje, cmap="gray")
plt.show()
