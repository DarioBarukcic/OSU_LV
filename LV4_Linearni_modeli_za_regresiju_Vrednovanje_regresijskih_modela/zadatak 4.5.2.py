import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import math
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import max_error

data = pd.read_csv('data_C02_emission.csv')

Y=data["CO2 Emissions (g/km)"]
X=data[["Engine Size (L)",
        "Cylinders", 
        "Fuel Consumption City (L/100km)",
        "Fuel Consumption Hwy (L/100km)", 
        "Fuel Consumption Comb (L/100km)", 
        "Fuel Consumption Comb (mpg)"]]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=1)

ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(data[["Fuel Type"]]).toarray()

data["Fuel Type"] = X_encoded

linearModel = lm.LinearRegression()
linearModel.fit(X_train, Y_train)

Y_test_p = linearModel.predict(X_test)

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

MAX_ERROR = max_error(Y_test, Y_test_p)

data["difference"] = abs(data['CO2 Emissions (g/km)'] - MAX_ERROR)
car = data[data["difference"] == data["difference"].max()]

print("Najveće odstupanje iznosi:", round(data["difference"].max(), 2), "g/km")
print("Najveće odstupanje je za automobil:")
print(car)
