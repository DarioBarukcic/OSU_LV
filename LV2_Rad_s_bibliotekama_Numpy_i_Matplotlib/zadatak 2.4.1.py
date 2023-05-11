import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,3,3,2,1])
y = np.array([1,1,2,2,1])

plt.plot(x,y, linewidth=1.5,marker=".", markersize=10, color="blue")
plt.axis([0.0, 4.0, 0.0, 4.0])
plt.xlabel("x os")
plt.ylabel("y os")
plt.title("Primjer")
plt.show()