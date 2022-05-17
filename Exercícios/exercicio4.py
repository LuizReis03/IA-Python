# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:50:44 2022

@author: DISRCT
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

dados=load_breast_cancer()
cancer=pd.DataFrame(data=dados.data, columns=dados.feature_names) 
cancer['Class']=dados.target 

x = cancer.iloc[:,:-1]
y = cancer.iloc[:,-1:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=30, shuffle=True)

print("\n\nTamanho do Dataset Treino {} Amostras".format(len(x_train)))
print("Benigno' {} amostras | Maligno {} Amostras".format(y_train.value_counts()[0], y_train.value_counts()[1]))

print("\n\nTamanho do Dataset Teste {} Amostras".format(len(x_test)))
print("Benigno {} amostras | Maligno {} Amostras\n\n".format(y_test.value_counts()[0], y_test.value_counts()[1]))

## modelo a ser utilizado
from sklearn import svm

model = svm.LinearSVC(max_iter=10000, random_state=9)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

#GERA MATRIX CONFUSAO
from sklearn.metrics import confusion_matrix
import seaborn

cm = confusion_matrix(y_test, y_pred)
print("MATRIX CONFUSÃO: \n", cm)

# obter contagens de observações
vdd, nf, f, nVdd = cm.ravel()

group_names = ['VERDADEIRO','FALSO','NÃO FALSO','NÃO VERDADEIRO']
group_counts = ['{0:0.0f}'.format(value) for value in
                cm.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cm.flatten()/np.sum(cm)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
seaborn.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False)

plt.show()

#APRESENTA OS DADOS DO GRÁFICO (acuracia, precisao, recall, f1)
accuracy = (vdd+f)/(vdd + nVdd + nf + f)
print("\nAcurácia de {0:0.2f}%".format(accuracy*100))
     
precision = (vdd)/(vdd+f)
print("Precisão de {0:0.0f}%".format(precision*100))

recall = (vdd)/(vdd + nf)
print("Recall de {0:0.2f}%".format(recall*100))

f1 = (2*precision*recall)/(recall+precision)
print("F1 - Score de {0:0.2f}%".format(f1*100))


#GRÁFICO DE REGIÃO DE DECISÃO
from mlxtend.plotting import plot_decision_regions
svc = SVC() 

clf = SVC(C=100, gamma=0.0001)
pca = PCA(n_components = 2)
y_train = np.ravel(y_train)
x_train = pca.fit_transform(x_train)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_train)
cm = confusion_matrix(y_train, y_pred)
print("MATRIX CONFUSÃO: \n", cm)

# obter contagens de observações
vdd, nf, f, nVdd = cm.ravel()

group_names = ['VERDADEIRO','FALSO','NÃO FALSO','NÃO VERDADEIRO']
group_counts = ['{0:0.0f}'.format(value) for value in
                cm.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cm.flatten()/np.sum(cm)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
seaborn.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False)

plt.show()

accuracy = (vdd+f)/(vdd + nVdd + nf + f)
print("\nAcurácia de {0:0.2f}%".format(accuracy*100))
     
precision = (vdd)/(vdd+f)
print("Precisão de {0:0.0f}%".format(precision*100))

recall = (vdd)/(vdd + nf)
print("Recall de {0:0.2f}%".format(recall*100))

f1 = (2*precision*recall)/(recall+precision)
print("F1 - Score de {0:0.2f}%".format(f1*100))

plot_decision_regions(x_train, y_train, clf=clf, legend=2)
plt.show()














