# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 11:12:48 2022

@author: DISRCT
"""

import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

dados = {'Densidade': [22.02,23.83,26.67,25.38,25.49,23.50,25.90,24.89,
         21.49,22.67,24.62,24.18,22.78,22.56,24.46,23.79,
         20.33,21.67,24.67,22.45,22.29,21.95,20.49,21.81]} 

plt.hist(dados['Densidade'])
plt.show()

#calculando media e desvio padrao
desvio_padrao = np.std(dados['Densidade'], ddof=1)
media = np.mean(dados['Densidade'])

#vizualizando a curva da distruibuicao normal
dominio = np.linspace(np.min(dados['Densidade']), np.max(dados['Densidade']))
plt.plot(dominio,norm.pdf(dominio, media, desvio_padrao))