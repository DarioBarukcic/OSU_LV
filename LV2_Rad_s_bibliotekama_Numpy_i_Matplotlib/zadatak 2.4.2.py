import numpy as np
import matplotlib.pyplot as plt

polje = np.loadtxt("data.csv",skiprows=1,delimiter=",")

visina = polje[:,1]
masa = polje[:,2]

print(f"Ukupno je {len(visina)} mjerenja")

plt.scatter(visina,masa, s=5)
plt.show()

visina2 = visina[::50]
masa2 = masa[::50]

plt.scatter(visina2,masa2, s=5)
plt.show()

print(f"maksimalna visina je {np.max(visina)}")
print(f"minimalna visina je {np.min(visina)}")
print(f"srednja visina je {np.mean(visina)}")

muski = polje[np.where(polje[:,0] == 1)]
zenski = polje[np.where(polje[:,0] == 0)]
muski = muski[:,1]
zenski = zenski[:,1]

print(f"Max visina muskaraca= {np.max(muski)}")
print(f"Min visina muskaraca= {np.min(muski)}")
print(f"Srednja visina muskaraca= {np.mean(muski)}")

print(f"Max visina zena= {np.max(zenski)}")
print(f"Min visina zena= {np.min(zenski)}")
print(f"Srednja visina zena= {np.mean(zenski)}")