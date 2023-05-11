import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, accuracy_score

X, Y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

#a)
plt.scatter(X_train[:,0], X_train[:,1], marker=".", c=Y_train, s=15, cmap=mcolors.ListedColormap(["red", "blue"]))
plt.scatter(X_test[:,0], X_test[:,1], marker="x", c=Y_test, s=25, cmap=mcolors.ListedColormap(["red", "blue"]))
#plt.show()

#b)
logisticRegression = LogisticRegression()
logisticRegression.fit(X_train,Y_train)

#c)
theta = logisticRegression.intercept_
coefs = logisticRegression.coef_.T

print(theta)
print(coefs)

a = -coefs[0]/coefs[1]
c = -theta/coefs[1]

xymin, xymax = -4, 4
xd = np.array([xymin, xymax])
yd = a*xd + c
plt.plot(xd, yd, linestyle='--')
plt.show()

#d)
Y_test_p = logisticRegression.predict(X_test)

cm = confusion_matrix(Y_test,Y_test_p)
print("Matrica zabune", cm)
disp = ConfusionMatrixDisplay(confusion_matrix(Y_test,Y_test_p))
disp.plot()
plt.show()

print('Točnost:', round(accuracy_score(Y_test, Y_test_p), 2))
print('Preciznost:', round(precision_score(Y_test, Y_test_p), 2))
print('Odziv:', round(recall_score(Y_test, Y_test_p), 2))

#e)
trueClasified = (Y_test==Y_test_p)
plt.scatter(X_test[:,0], X_test[:,1], c=trueClasified, s=25, cmap=mcolors.ListedColormap(["red", "green"]))
plt.show()
