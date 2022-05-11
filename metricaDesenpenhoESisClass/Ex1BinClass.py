# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 13:21:19 2020

@author: OLV2CT
"""
#PROGRAMA PARA BALANCEAMENTO DE CLASSES 

#import io
#import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ignorar warnings
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("dados_exe1.csv", header=0)
print(df.head())
#print(df["famhist"])

# Codificar valores categóricos da coluna "famhist" para inteiros
df["famhist"] = df["famhist"].astype('category')
#print(df["famhist"])

#Separa os dados em uma lista finita com duas categorias
df["famhist"] = df["famhist"].cat.codes
#print(df["famhist"])

print(df.head())
#Cria cada item da lista com um código

# separar dados
y = df.iloc[:,-1]
x = df.iloc[:,:-1] #Pega todas as colunas menos a última

print("Tamanho do Dataset Completo {} Amostras".format(len(x)))
print("Classe '0' {} amostras | Classe '1' {} Amostras".format(y.value_counts()[0], y.value_counts()[1]))
print("")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=20, shuffle=True)

print("Tamanho do Dataset Treino {} Amostras".format(len(x_train)))
print("Classe '0' {} amostras | Classe '1' {} Amostras".format(y_train.value_counts()[0], y_train.value_counts()[1]))

print("Tamanho do Dataset Teste {} Amostras".format(len(x_test)))
print("Classe '0' {} amostras | Classe '1' {} Amostras".format(y_test.value_counts()[0], y_test.value_counts()[1]))
# DEPOIS DE SEPADADO E BALANCEADO ELE ESTÁ APTO PARA SER TREINADO


## modelo a ser utilizado
from sklearn import svm

model = svm.LinearSVC(max_iter=10000, random_state=0)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
import seaborn

cm = confusion_matrix(y_test, y_pred)

# obter contagens de observações
tn, fp, fn, tp = cm.ravel()
#
# Visualização da Matriz de Confusão
group_names = ['TN','FP','FN','TP']
group_counts = ['{0:0.0f}'.format(value) for value in
                cm.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cm.flatten()/np.sum(cm)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
seaborn.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False)

print("Confusion Matrix : ")

accuracy = (tp+tn)/(tp+fp+tn+fn)
print("Acurácia de {0:0.2f}%".format(accuracy*100))
     
precision = (tp)/(tp+fp)
print("Precisão de {0:0.0f}%".format(precision*100))

recall = (tp)/(tp+fn)
print("Recall de {0:0.2f}%".format(recall*100))

f1 = (2*precision*recall)/(recall+precision)
print("F1 - Score de {0:0.2f}%".format(f1*100))
#
#      
#
#
#
#
#
#
#
#
