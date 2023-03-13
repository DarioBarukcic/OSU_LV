import numpy as np
import matplotlib.pyplot as plt

polje = np.loadtxt("data.csv", skiprows=1, delimiter=",")
# print(polje)

visina = polje[:, 1]
masa = polje[:, 2]

print(f"Ukpno je {len(visina)} mjerenja")

plt.scatter(visina, masa, s=5)
plt.show()

visina2 = visina[::50]
masa2 = masa[::50]

plt.scatter(visina2, masa2, s=5)
plt.show()

print(f"Max visina= {np.max(visina)}")
print(f"Min visina= {np.min(visina)}")
print(f"Srednja visina= {np.mean(visina)}")

muski = []
zenski = []

for row in polje:
    if (row[0] == 1):
        muski.append(row[1])
    else:
        zenski.append(row[1])

print(f"Max visina muskaraca= {np.max(muski)}")
print(f"Min visina muskaraca= {np.min(muski)}")
print(f"Srednja visina muskaraca= {np.mean(muski)}")

print(f"Max visina zena= {np.max(zenski)}")
print(f"Min visina zena= {np.min(zenski)}")
print(f"Srednja visina zena= {np.mean(zenski)}")
