import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data_C02_emission.csv")

#data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
#                     columns= iris['feature_names'] + ['target'])

#a)
print(f"broj mjerenja je {len(data)}")
print(f"Tipovi podataka {data.info()}")
print(f"Postoji {data.isnull().sum()} podataka sa izostavljenim vrijednostima")
data.dropna(axis=0)
print(f"Postoji {data.duplicated().sum()} duplih podataka")
data.drop_duplicates()
data = data.reset_index(drop = True)
for collum in ["Make", "Model","Vehicle Class", "Transmission", "Fuel Type"]:
    data[collum] = data[collum].astype("category")

#b)
sorted = data.sort_values(by=["Fuel Consumption City (L/100km)"])
print("Najveca potrosnja u gradu:\n", sorted[["Make","Model","Fuel Consumption City (L/100km)"]].tail(3))
print("Najmanja potrosnja u gradu:\n", sorted[["Make","Model","Fuel Consumption City (L/100km)"]].head(3))

#c)
engineSizeInterval = data[(data["Engine Size (L)"]>2.5) & (data["Engine Size (L)"]<3.5)]
print("Broj vozila izmedu 2.5 i 3.5 l motorima: ", len(engineSizeInterval))
print("Prosječna emisija CO2 ovih vozila je:", engineSizeInterval["CO2 Emissions (g/km)"].mean())

#d)
audiCars = data[data["Make"]=="Audi"]
print("audi proizvoid ", len(audiCars) ," auta")
audi4Cylinder = audiCars[audiCars["Cylinders"]==4]
print("Prosjecna emisija CO2 audijevih auta sa 4 clindra je ", audi4Cylinder["CO2 Emissions (g/km)"].mean())

#e)
grouped_by_cilinders_count = data.groupby("Cylinders").count()
print(grouped_by_cilinders_count["Make"])


#f)
dizel = data[data["Fuel Type"]== "D"]
gasoline = data[data["Fuel Type"]== "X"]

print("Prosječna potrošnja dizelaša je", round(dizel["Fuel Consumption City (L/100km)"].mean(),2), "L/100km")
print("Prosječna potrošnja benzinaca je", round(gasoline["Fuel Consumption City (L/100km)"].mean(),2), "L/100km")

print("Medijalna potrošnja dizelaša je", round(dizel["Fuel Consumption City (L/100km)"].median(), 2), "L/100km")
print("Medijalna potrošnja benzinaca je", round(gasoline["Fuel Consumption City (L/100km)"].median(), 2), "L/100km")

#g)
matchingConditions = data[(data["Fuel Type"]== "D") & (data["Cylinders"]==4)]
maxConsumption = matchingConditions.sort_values(by="Fuel Consumption City (L/100km)")
print("Najveću gradsku potrošnju ima auto sa 4 cilindra ima:\n ",maxConsumption.tail(1))

#h)
manualTransmission = data[data["Transmission"].str.startswith("M")]
print("Postoji", len(manualTransmission), "vozila s ručnim mjenjačem.")

#i)
print(data.corr(numeric_only=True))