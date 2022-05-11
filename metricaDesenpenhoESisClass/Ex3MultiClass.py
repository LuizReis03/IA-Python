# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 16:30:07 2020

@author: olv2ct
"""

# Carregar Pacotes

#import io
#import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ignorar warnings
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("iris.csv", header=0)
#df.head() # Tamanho ta setosa, pétala,classe

x= df.iloc[:,:-1]
y= df.iloc[:, -1]


labels=["Setosa", "Versicolor", "Virginica"]
print("Tamanho do Dataset Completo {} amostras".format(len(x)))
print("{} {} amostras, {} {} amostras, {} {} amostras".format(
        labels[0], y.value_counts()[0],
        labels[1], y.value_counts()[1],
        labels[2], y.value_counts()[2]))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=20, shuffle=True)

print("\n\nTamanho do Dataset Treino {} Amostras".format(len(x_train)))
print("Classe '0' {} amostras | Classe '1' {} Amostras".format(y_train.value_counts()[0], y_train.value_counts()[1]))

print("\n\nTamanho do Dataset Teste {} Amostras".format(len(x_test)))
print("Classe '0' {} amostras | Classe '1' {} Amostras".format(y_test.value_counts()[0], y_test.value_counts()[1]))
# DEPOIS DE SEPADADO E BALANCEADO ELE ESTÁ APTO PARA SER TREINADO

## modelo a ser utilizado
from sklearn import svm

model = svm.LinearSVC(max_iter=10000, random_state=9)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
import seaborn

cm = confusion_matrix(y_test, y_pred)

# obter contagens de observações
tse, fvs, fvi, fse, tvs, fvi, fse, fvs, tvi  = cm.ravel()

# Visualização da Matriz de Confusão
group_names = ['TSE','FVS','FVI','FSE', 'TVS', 'FVI', 'FSE', 'FVS', 'TVI']
group_counts = ['{0:0.0f}'.format(value) for value in
                cm.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cm.flatten()/np.sum(cm)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(3,3)
seaborn.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False)

print("\n\nConfusion Matrix : ")

accuracy = (tse+tvs+tvi)/(tse + fvs + fvi + fse + tvs + fvi + fse + fvs + tvi)
print("Acurácia de {0:0.2f}%".format(accuracy*100))
     
precision = (tse)/(tse+fvs+fvi)
print("Precisão de {0:0.0f}%".format(precision*100))

recall = (tse)/(tse + fse + fse)
print("Recall de {0:0.2f}%".format(recall*100))

f1 = (2*precision*recall)/(recall+precision)
print("F1 - Score de {0:0.2f}%".format(f1*100))












