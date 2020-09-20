import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing  import PolynomialFeatures
from sklearn import linear_model

## Carga de datos ##
data = pd.read_csv('covid_cases.csv', sep = ',')
data = data [['id', 'cases']]
print('*'*30); print('data.head()'); print('*'*30)
print (data.head())

## Preparación de los datos ##
print('*'*30); print('Prepocesamiento de los Datos'); print('*'*30)
x = np.array(data['id']).reshape(-1, 1)
y = np.array(data['cases']).reshape(-1, 1)
plt.plot(y, '-m')

polyFeat = PolynomialFeatures(degree=3)
x = polyFeat.fit_transform(x)

## Entrenar el modelo ##
print('*'*30); print('Entrenar el modelo'); print('*'*30)
model = linear_model.LinearRegression()
model.fit(x, y)
accuracy = model.score(x,y)
print(f'Acccuracy: { round(accuracy*100,3)} %')
y0 = model.predict(x)

## Predecir ##
days = 30
print('*'*30); print('Predicción'); print('*'*30)
print(f'Predicción - Casos luego de {days} días')
prediccion = int(model.predict(polyFeat.fit_transform([[264+days]])))
print(round(prediccion/1000000, 2), 'Millones')

x1 = np.array(list(range(1, 264+days))).reshape(-1, 1)
y1 = model.predict(polyFeat.fit_transform(x1))
plt.plot(y1, '--r')
plt.plot(y0, '--b')
plt.show()