import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("data_C02_emission.csv")

#a)
plt.figure ()
data ["CO2 Emissions (g/km)"].plot(kind="hist", bins=20)
plt.xlabel("Interval of CO2 emissions (g/km)")
plt.ylabel("Noumber of cars in category")
plt.title("CO2 Emissions (g/km)")
plt.show()

#b)
colors = {"X":"green","Z":"red", "D":"blue", "E":"black", "N":"yellow"}
data.plot.scatter(x="Fuel Consumption City (L/100km)",
                    y="CO2 Emissions (g/km)",
                    c=data["Fuel Type"].apply(lambda x:colors[x]), s=5)
plt.show()

#c)
grouped_fuel_type = data.groupby("Fuel Type")
grouped_fuel_type.boxplot(column=['Fuel Consumption Hwy (L/100km)'])
plt.show()

#d)
grouped_fuel_type["Make"].count().plot(kind="bar")
plt.show()

#e)
grouped_Cylinders = data.groupby("Cylinders")
grouped_Cylinders["CO2 Emissions (g/km)"].mean().plot(kind="bar")
plt.show()