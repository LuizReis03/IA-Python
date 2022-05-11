# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 08:26:50 2020

@author: DISRCT
"""


from __future__ import division #Para evitar confundir as ferramentas existentes que analisam declarações de importação e esperam encontrar os módulos que estão importando.
from random import random
from math import pi
import matplotlib.pyplot as plt


def rain_drop(length_of_field=1):
    return [(.5 - random()) * length_of_field, (.5 - random()) * length_of_field]
#Crio um valor para a gota de chuva que respeite uma distribuição uniforme, Um número aleatório 
#multiplica pelo tamanho da sua area

def is_point_in_circle(point, length_of_field=1):
    return (point[0]) ** 2 + (point[1]) ** 2 <= (length_of_field / 2) ** 2
#Função para validar se esse ponto se encontra dentro do circulo

def plot_rain_drops(drops_in_circle, drops_out_of_circle, length_of_field=1):
    plt.figure()
    plt.xlim(-length_of_field / 2, length_of_field / 2)
    plt.ylim(-length_of_field / 2, length_of_field / 2)
    plt.scatter([e[0] for e in drops_in_circle], [e[1] for e in drops_in_circle], color='blue', label="Gotas no "
                                                                                                      "Círculo")
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.title("N Gostas Dentro do Círculo")
    plt.scatter([e[0] for e in drops_out_of_circle], [e[1] for e in drops_out_of_circle], color='black',
                label="Gotas Fora do Círculo")
    plt.legend(loc="center")

def rain(number_of_drops=100000, length_of_field=1):
    number_of_drops_in_circle = 0
    drops_in_circle = []
    drops_out_of_circle = []
    pi_estimate = []
    for k in range(number_of_drops):
        d = (rain_drop(length_of_field))
        if is_point_in_circle(d, length_of_field):
            drops_in_circle.append(d)
            number_of_drops_in_circle += 1
        else:
            drops_out_of_circle.append(d)
        pi_estimate.append(
            4 * number_of_drops_in_circle / (k + 1))  # Insere na lista os valores de pi estimado
    
    plot_rain_drops(drops_in_circle, drops_out_of_circle, length_of_field)
    plt.figure()
    plt.scatter(range(1, number_of_drops + 1), pi_estimate)
    max_x = plt.xlim()[1]
    plt.hlines(pi, 0, max_x, color='black')
    plt.xlim(0, max_x)
    plt.title("$\pi$ Real vs. Estimado")
    plt.xlabel("N")
    plt.ylabel("$\pi$")
# Defino o número de gotas que irão cair, gerando os pontos das gotas aleatórias

    return  4 * (number_of_drops_in_circle / number_of_drops)
#No final apresenta o número aproximado de pi






#Definir pi com N= 100
	
#N = 100
#r = rain(N)
#print(f"Valor Estimado de pi = {r:0.8f}")
#print(f"Valor de pi Real {pi:0.8f}")



#
#N = 1000
#r = rain(N)
#print(f"Valor Estimado de pi = {r:0.8f}")
#
#N = 10000
#r = rain(N)
#print(f"Valor Estimado de pi = {r:0.8f}")


import time
N = 10000

start = time.time()
r = rain(N)
print(f"Valor Estimado de pi = {r:0.8f}")
print("elapsed time {0:0.2f}s".format(time.time()-start))
print("erro:",abs(pi - r))