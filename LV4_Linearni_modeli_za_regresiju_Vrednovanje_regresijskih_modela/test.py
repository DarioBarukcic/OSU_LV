#import bibilioteka
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from tensorflow import keras
from keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#učitavanje dataseta
iris = datasets.load_iris()
#data = datasets.load_iris()

#df = pd.DataFrame(data=data.data, columns=data.feature_names)

data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])


##################################################
#1. zadatak
##################################################


#a)
versicolour_data = (data1[ data1['target'] == 1])
plt.scatter(versicolour_data[['sepal length (cm)']], versicolour_data[['petal length (cm)']], c = "red", label="versicolour")
plt.title('odnos duljine latice i cašice svih pripadnika klase versicolour')
plt.show()
virginica_data = (data1[ data1['target'] == 2])
plt.scatter(versicolour_data[['sepal length (cm)']], versicolour_data[['petal length (cm)']], c = "red", label="versicolour")
plt.scatter(virginica_data[['sepal length (cm)']], virginica_data[['petal length (cm)']], c = "blue", label="virginica")
plt.title('odnos duljine latice i cašice svih pripadnika klase versicolour i virginica')
plt.xlabel("sepal length (cm)")
plt.ylabel("petal length (cm)")
plt.legend(loc="upper left")
plt.show()

#b)
grouped = data1.groupby('target').mean()
grouped["sepal width (cm)"].plot(kind="bar")
plt.title("prosječna vrijednost širine cašice za sve tri klase")
plt.xlabel("klase")
plt.ylabel("prosječna širina čašice")
plt.show()

#c)
virginica_avg = virginica_data["sepal width (cm)"].mean()
count = virginica_data[virginica_data["sepal width (cm)"] > virginica_avg].count()
print(count)


##################################################
#2. zadatak
##################################################
#a) i b)
input_variables = ["sepal width (cm)", 'petal width (cm)', 'sepal length (cm)', 'petal length (cm)']


#b)
X = data1[input_variables].to_numpy()
Js = []
Ks = range(1, 8)
for i in range(1, 8):
    km = KMeans(n_clusters=i, init='random', n_init=5, random_state=0)

    km.fit(X)
    Js.append(km.inertia_)
    labels = km.predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=km.labels_.astype(float))
    plt.title(f'K={i}')
    plt.show()
plt.plot(Ks, Js)
plt.show()

#c)
km = KMeans(n_clusters=3, init='random', n_init=5, random_state=0)
km.fit(X)
labels = km.predict(X)


#d)
plt.scatter(X[:, 0], X[:, 1], c=km.labels_.astype(float))
centroids = km.cluster_centers_
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]
plt.scatter(centroids_x,centroids_y,marker = "x", s=50,linewidths = 5, zorder = 10, c='red')
plt.title("Dijagramom raspršenja ")
plt.xlabel("sepal width (cm)")
plt.ylabel("petal width (cm)")
plt.legend(['Setosa','centroidi', 'Virginica'], loc='upper left')
plt.show()
#e)
score = metrics.accuracy_score(data1['target'],labels)
print("Tocnost klasifikacije je: ", score)
##################################################
#3. zadatak
##################################################

#predobrada podataka
input_variables = ["sepal width (cm)", 'petal width (cm)', 'sepal length (cm)', 'petal length (cm)']
output_variables = ['target']
X = data1[input_variables].to_numpy()
y = data1[output_variables].to_numpy()

(X_train, X_test, y_train, y_test) = train_test_split(
    X, y, test_size=0.25, random_state=1)

sc = StandardScaler()
X_train_n = pd.DataFrame(sc.fit_transform(X_train))
X_test_n = pd.DataFrame(sc.transform(X_test))
#a)


model = keras.Sequential()
model.add(layers.Input(shape=(4,)))
model.add(layers.Dense(10, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(7, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(5, activation="relu"))
model.add(layers.Dense(3, activation="softmax"))
model.summary()

#b)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy",])
#c)
history = model.fit(X_train, y_train, batch_size=7, epochs=500, validation_split=0.1)
#d)
keras.models.save_model(model, "model")
#e)
model = keras.models.load_model("model")
score = model.evaluate(X_test, y_test, verbose=0)
#f)
y_test_pred = model.predict(X_test)             
y_test_pred = np.argmax(y_test_pred, axis=1)   

cm = confusion_matrix(y_test, y_test_pred)
print("Matrica zabune:", cm)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred))
disp.plot()
plt.show()