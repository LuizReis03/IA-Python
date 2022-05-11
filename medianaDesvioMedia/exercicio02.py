# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 09:29:10 2022

@author: DISRCT
"""

import pandas as pd
import numpy as np

dados = {'Mistura 1': [22.02,23.83,26.67,25.38,25.49,23.50,25.90,24.89],
         'Mistura 2': [21.49,22.67,24.62,24.18,22.78,22.56,24.46,23.79],
         'Mistura 3': [20.33,21.67,24.67,22.45,22.29,21.95,20.49,21.81]} 

df_mistura = pd.DataFrame(data=dados)

media1 = df_mistura['Mistura 1'].mean()
mediana1 = df_mistura['Mistura 1'].median()
desvio1 = df_mistura['Mistura 1'].std()

media2 = df_mistura['Mistura 2'].mean()
mediana2 = df_mistura['Mistura 2'].median()
desvio2 = df_mistura['Mistura 2'].std()

media3 = df_mistura['Mistura 3'].mean()
mediana3 = df_mistura['Mistura 3'].median()
desvio3 = df_mistura['Mistura 3'].std()

print("Media mistura 1: ", round(media1,2))
print("Mediana mistura 1: ", round(mediana1,2))
print("Desvio padrão mistura 1: ", round(desvio1,2))

print("\n\nMedia mistura 2: ", round(media2,2))
print("Mediana mistura 2: ", round(mediana2,2))
print("Desvio padrão mistura 2: ", round(desvio2,2))

print("\n\nMedia mistura 3: ", round(media3,2))
print("Mediana mistura 3: ", round(mediana3,2))
print("Desvio padrão mistura 3: ", round(desvio3,2))

print("\n", df_mistura.describe())

df_mistura.boxplot(column=['Mistura 1', 'Mistura 2', 'Mistura 3'], grid=False)