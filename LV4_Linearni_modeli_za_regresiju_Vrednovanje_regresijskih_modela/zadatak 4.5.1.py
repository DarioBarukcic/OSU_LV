import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import math
from sklearn.metrics import r2_score

data = pd.read_csv('data_C02_emission.csv')

#a)
Y=data["CO2 Emissions (g/km)"]
X=data[["Engine Size (L)","Cylinders", "Fuel Consumption City (L/100km)","Fuel Consumption Hwy (L/100km)", "Fuel Consumption Comb (L/100km)", "Fuel Consumption Comb (mpg)"]]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=1)

#b)
plt.scatter(X_train["Fuel Consumption City (L/100km)"],Y_train, c='b', label='Train', s=1)
plt.scatter(X_test["Fuel Consumption City (L/100km)"],Y_test, c='r', label='Test', s=1)
plt.xlabel("Fuel Consumption City (L/100km)")
plt.ylabel("CO2 Emissions (g/km)")
plt.legend()
plt.show()

#c)
plt.figure()
plt.hist(X_train["Fuel Consumption City (L/100km)"])

sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
plt.figure()
plt.hist(X_train_n[:,2])
plt.show()
X_test_n = sc.transform(X_test)

#d)
linearModel = lm.LinearRegression()
linearModel.fit(X_train_n , Y_train)
print(f'Parametri modela:{linearModel.coef_}')

#e)
Y_test_p = linearModel.predict(X_test_n)

plt.scatter(Y_test,Y_test_p, s=1)
plt.show()

#f)
MAE = mean_absolute_error(Y_test,Y_test_p)
print(f"MAE: {MAE}")

MSE = mean_squared_error(Y_test,Y_test_p)
print(f"MSE: {MSE}")

RMSE = math.sqrt(MSE)
print(f"RMSE: {RMSE}")

MAPE = mean_absolute_percentage_error(Y_test,Y_test_p)
print(f"MAPE: {MAPE}")

R2 = r2_score(Y_test, Y_test_p)
print(f"R2: {R2}")

#g)
#Vrijednosti rastu sto upucuje na do da je model tocniji ako ima vise ulaznih parametara prema kojima ce napraviti model.