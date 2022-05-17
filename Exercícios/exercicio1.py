# -*- coding: utf-8 -*-
"""
Created on Mon May  2 10:26:28 2022

@author: DISRCT
"""

# Manipulação de dados
import pandas as pd
import numpy as np
import math

# Gráficos
import matplotlib.pyplot as plt
import seaborn as sns


# Algoritmos, Métricas e Funcionalidades de Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


#Gerando o dataset
dados = pd.read_csv('weatherHistory.csv')
dados = pd.DataFrame(data=dados)

# separar dados
y = dados["Temperature (C)"]
x = dados["Humidity"]

X_train, X_test, y_train, y_test=train_test_split(x,y, test_size=0.25, random_state=13)

#Regressão Linear
sns.scatterplot(X_train, y_train, color='blue', label='Dados de treino')
sns.scatterplot(X_test, y_test, color='green', label='Dados de test')
plt.legend()
plt.show()

linreg=LinearRegression()

X_trainOut = X_train.values.reshape(-1,1)
X_testOut = X_test.values.reshape(-1,1)

linreg.fit(X_trainOut, y_train)
y_pred_ = linreg.predict(X_trainOut)
print('R2 de treino:{:.2f}'.format(r2_score(y_train, y_pred_)))
print('R2 de teste:{:.2f}'.format(r2_score(y_test, linreg.predict(X_testOut))))

sns.scatterplot(X_train, y_train, color='blue', label='Dados de treino')
sns.scatterplot(X_test, y_test, color='green', label='Dados de test')
sns.lineplot(X_train, y_pred_, color='red')
plt.legend()
plt.show()

# Observando os coeficientes da função
print('Nosso [B0]: ', linreg.intercept_)
print('Nosso [B1]: ', linreg.coef_)
print('Nossa função gerada foi: Y = {} + {}*X'.format(linreg.intercept_, linreg.coef_[0]))
print("MAE treino: {:.0f}%".format(mean_absolute_error(y_train, linreg.predict(X_trainOut))))
print("MAE teste: {:.0f}%".format(mean_absolute_error(y_test, linreg.predict(X_testOut))))
print("RMSE treino: {:.0f}%".format(math.sqrt(mean_squared_error(y_train, linreg.predict(X_trainOut)))))
print("RMSE teste: {:.0f}%".format(np.sqrt(mean_squared_error(y_test, linreg.predict(X_testOut)))))

humidade_pred = np.array(0)
previsao = linreg.predict(humidade_pred.reshape(-1,1))
formata = round(previsao[0],2)
print("\n\nHumidade {} = Temperatura {}°".format(humidade_pred, formata))

