# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 11:34:35 2022

@author: DISRCT
"""

import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import pandas as pd

dados = {'Idades': [14,17,18,15,15,16,17,15,16,16,15,17,15,16,
         16,18,18,19,17,16,17,15,16,17,17,19,20,18,
         17,16,15,16,16,17,18,18,17,17,15,16,16,15]}

df = pd.DataFrame(data=dados)

mediaNum = df['Idades'].mean()
desvioNum = df['Idades'].std()

print("Media:",round(mediaNum,2))
print("Desvio padrão:",round(desvioNum,2))

#calculando media e desvio padrao
desvio_padrao = np.std(dados['Idades'], ddof=1)
media = np.mean(dados['Idades'])
dominio = np.linspace(np.min(dados['Idades']), np.max(dados['Idades']))

df.boxplot(column=['Idades'], grid=False)
plt.show()
plt.hist(dados['Idades'])
plt.show()
plt.plot(dominio,norm.pdf(dominio, media, desvio_padrao))
plt.show()