# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 07:31:47 2020

@author: OLV2CT
"""

# Manipulação de dados
import pandas as pd
import numpy as np
# Dataset Iris (pode ser baixado manualmente e carregado pelo pd.read_csv() ou direto pelo sklearn.datasets)
from sklearn.datasets import load_iris
# Algoritmos, Métricas e Funcionalidades de Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC #classificador
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#Utilizando a função load_iris do sklearn iremos ter acesso a um dicionário com as informações do dataset
iris = load_iris() #Este é um dataset de flores
print(iris.keys())
# Classes
print('Classes/Rótulos: ', iris.target_names)
for i, value in enumerate(iris.target_names):
    print(f'{i} : {value}')
    
## Os dados já estão em formato np.array (ideais para inserir nos algoritmos do sklearn)
## Mas para entendimento iremos converter em um dataframe
features = pd.DataFrame(iris.data, columns=iris.feature_names)
target = pd.DataFrame(iris.target, columns=['target'])
## Observando dataset completo
dataset = pd.concat([features, target], axis=1)
# Vamos pegar 10 linhas aleatórias e observar os dados 
print(dataset.sample(10))
## Verificando se existem nulos
print(dataset.isna().sum())
## Observando informações estastíticas gerais
print(dataset.describe())
## Podemos observar a correlação das váriaveis
## Para criação do modelo queremos features altamente correlacionadas com o target
## Quando temos diversas váriaveis também e comum retirar as colunas de features que estão muito relacionadas 
## (não desejamos colinearidade)
## Podemos observar a relação entre as váriaveis
sns.pairplot(dataset, hue='target')
plt.show()
## Vamos escolher 2 features para serem utilizadas para modelagem
print('Características/Features possíveis: ', iris.feature_names)
## Para selecionar e plotar outras features apenas substitua os valores abaixo. 
## Neste exemplo usaremos sepal width e petal lenght que parecem ser altamente separáveis no gráfico acima
## Fica a cargo de cada um testar outras váriaveis e rodar novamente as células
feat1 = 'sepal width (cm)'
feat2 = 'petal length (cm)'
print('Features escolhidas: {} e {}'.format(feat1, feat2))
sns.scatterplot(features[feat1], features[feat2], hue=target['target'], palette='brg')
plt.title(feat1 + ' x ' + feat2)
plt.show()
## A maioria dos métodos do sklearn aceitam pandas.DataFrame, pandas.Series e np.array
## A saída sempre será em np.array
## Recomenda-se trabalhar apenas com np.arrays, para isso aplica-se .values se os dados vierem de em formato do pandas 
X = features[[feat1,feat2]].values
y = target.values
print('Tipo de saída de X:{} e y:{}'.format(type(X), type(y)))
print('Shape de saída de X:{} e y:{}'.format(X.shape, y.shape))
## Divisão treino e teste
## Random State permite repetibilidade no processo (se tiver mesmo número iremos ter o mesmo resultado)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=11)

print('Quantidade de amostras de treino: {}'.format(X_train.shape[0]))
print('Quantidade de amostras de teste: {}'.format(X_test.shape[0]))
# Instanciando os algoritmos que serão utilizados
logreg = LogisticRegression()
y_train=y_train.astype(int)
# O algoritmo será treinado com os valores default dos parametros
# Os valores utilizados para cada algoritmo podem ser vistos abaixo da célula de treino
# ou através do método .get_params()
logreg.fit(X_train, y_train)
print(logreg.get_params())
##C é o parâmetro de regularização e podemos alterar a fronteira de decisão alterando o valor de C.
##Quanto menor o C, maior a regularização, menor a complexidade do modelo.
##Quanto maior o C, menor a regularização, aumentando complexidade do modelo.
##    
from mlxtend.plotting import plot_decision_regions
# plot_decision_regions(X,y,classificador)
plot_decision_regions(X_train, 
                      y_train.flatten(), #Flatten transforma a matriz de 105,1 para um array
                      logreg)
plt.show()
print('Score do LogisticRegression para dados de treino: {:.2%}'.format(logreg.score(X_train, y_train)))
print('Score do LogisticRegression para dados de teste: {:.2%}'.format(logreg.score(X_test, y_test)))
## Decision Tree
dtc = DecisionTreeClassifier()
## Árvore de decisão
dtc.fit(X_train, y_train)
print(dtc.get_params())
##O parâmetro max_depth é aquele que diz até onde vai a profundidade da árvore. Seu valor 
##default é None, ou seja, não limita a profundidade da árvore. Ao analisar a acurácia de 
##teste poderemos perceber que o algoritmo não obtem um resultado bom para generalização com
## este valor padrão. Deve-se testar valores diferentes a fim de se obter um balanceamento entre 
## treino e teste.
##    
## plot_decision_regions(X,y,classificador)
plot_decision_regions(X_train, 
                      y_train.flatten(), 
                      dtc)
plt.show()
print('Score do DecisionTreeClassifier para dados de treino: {:.2%}'.format(dtc.score(X_train, y_train)))
print('Score do DecisionTreeClassifier para dados de teste: {:.2%}'.format(dtc.score(X_test, y_test)))
#DT sempre funciona com cortes ortogonais
##algumas partes da classe laranja está na verde, por causa do overfitting
## KNN
knn = KNeighborsClassifier()
## Algoritmo KNN
## Default para vizinhos = 5
#
knn.fit(X_train, y_train)
#
print(knn.get_params())
#
##O parâmetro mais importante aqui é o de n_neighbours pois delimita a quantidade
## de vizinhos que irão definir o rótulo do dado novo. O valor deve ser ímpar, para 
## evitar que a votação tenha empate, e deve-se testar diferentes valores a fim de se obter 
## aquele com melhor balanceamento e com menor erro.
## plot_decision_regions(X,y,classificador)
plot_decision_regions(X_train, 
                      y_train.flatten(), 
                      knn)
plt.show()
print('Score do KNN para dados de treino: {:.2%}'.format(knn.score(X_train, y_train)))
print('Score do KNN para dados de teste: {:.2%}'.format(knn.score(X_test, y_test)))
## Instanciando os algoritmos que serão utilizados
svc = SVC()
# SVC
svc.fit(X_train, y_train)
print(svc.get_params())
##Três parâmetros importantes ao se utilizar o SVM:
##
##Escolha do Kernel: Linear para dados lineares, Polinomial e RBF para dados não lineares.
##C: Define grau de tolerância a erro, valores pequenos permitem margem maiores (generaliza melhor), valores grandes permitem menos erros e geram margem menores.
##Gamma: Usado apenas quando se tem o kernel RBF, valores mais altos geram modelos mais complexos (podem tender a overfitting).   
## plot_decision_regions(X,y,classificador)
plot_decision_regions(X_train, 
                      y_train.flatten(), 
                      svc)
plt.show()
print('Score do SVC para dados de treino: {:.2%}'.format(svc.score(X_train, y_train)))
print('Score do SVC para dados de teste: {:.2%}'.format(svc.score(X_test, y_test)))
#
## Comparando classificadores com valores default:
#
y_pred_lr = logreg.predict(X_test)
y_pred_dtc = dtc.predict(X_test)
y_pred_knn = knn.predict(X_test)
y_pred_svc = svc.predict(X_test)
#
print('\nMatriz confusão da Logistic Regression: \n')
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
#
print('\nMatriz confusão da Decision Tree: \n')
print(confusion_matrix(y_test, y_pred_dtc))
print(classification_report(y_test, y_pred_dtc))

print('\nMatriz confusão do KNN: \n')
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

print('\nMatriz confusão do SVC: \n')
print(confusion_matrix(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))

#    