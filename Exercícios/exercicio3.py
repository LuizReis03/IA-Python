# -*- coding: utf-8 -*-
"""
Created on Thu May 12 10:51:59 2022

@author: DISRCT
"""

import io
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Ignorar warnings
import warnings
warnings.filterwarnings('ignore')

dados = pd.read_csv('candy-data.csv', header=0)
y = dados.iloc[:,1]
x = dados.iloc[:, 2:]

labels=["Chocolate", "Não chocolate"]
print("{} {} amostras, {} {} amostras".format(
        labels[0], y.value_counts()[0],
        labels[1], y.value_counts()[1]))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=30, shuffle=True)

print("\n\nTamanho do Dataset Treino {} Amostras".format(len(x_train)))
print("Classe '0' {} amostras | Classe '1' {} Amostras".format(y_train.value_counts()[0], y_train.value_counts()[1]))

print("\n\nTamanho do Dataset Teste {} Amostras".format(len(x_test)))
print("Classe '0' {} amostras | Classe '1' {} Amostras".format(y_test.value_counts()[0], y_test.value_counts()[1]))

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
choVdd, nchoF, choF, nchoVdd = cm.ravel()

# Visualização da Matriz de Confusão
group_names = ['CHOCOLATE VERDADEIRO','CHOCOLATE FALSO','NÃO CHOCOLATE FALSO','NÃO CHOCOLATE VERDADEIRO']
group_counts = ['{0:0.0f}'.format(value) for value in
                cm.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cm.flatten()/np.sum(cm)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
seaborn.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False)

print("\n\nConfusion Matrix : ")

accuracy = (choVdd+choF)/(choVdd + nchoVdd + nchoF + choF)
print("Acurácia de {0:0.2f}%".format(accuracy*100))
     
precision = (choVdd)/(choVdd+choF)
print("Precisão de {0:0.0f}%".format(precision*100))

recall = (choVdd)/(choVdd + nchoF)
print("Recall de {0:0.2f}%".format(recall*100))

f1 = (2*precision*recall)/(recall+precision)
print("F1 - Score de {0:0.2f}%".format(f1*100))

model = svm.SVC(kernel='linear')
clf = model.fit(x, y)

fig, ax = plt.subplots()

from mlxtend.plotting import plot_decision_regions
svc = SVC() 

clf = SVC(C=100, gamma=0.0001)
pca = PCA(n_components = 2)
y_train = np.ravel(y_train)
x_train = pca.fit_transform(x_train)
clf.fit(x_train, y_train)

plot_decision_regions(x_train, y_train, clf=clf, legend=2)
plt.show()





