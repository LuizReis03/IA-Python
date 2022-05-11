# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 10:53:29 2022

@author: DISRCT
"""

import random 
import pandas as pd
lista=[random.randint(1,6) for i in range(1000)]
lista = pd.DataFrame(data=lista, columns=['Lado'])
lista.plot.hist(align="right", bins=6, rwidth=0.9)