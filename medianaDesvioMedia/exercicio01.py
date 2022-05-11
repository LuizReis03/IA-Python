# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 09:29:10 2022

@author: DISRCT
"""

import pandas as pd
import numpy as np

grupo_1 = (1,8,10,38,39)

print("Media: ", np.mean(grupo_1))
print("Mediana: ", np.median(grupo_1))
print("Desvio Padrão: ", np.std(grupo_1))

#PANDAS
dados = {'Grupo 1': [1,8,10,38,39],
      'Grupo 2':[8,10,39,49,45]}

df = pd.DataFrame(data=dados)