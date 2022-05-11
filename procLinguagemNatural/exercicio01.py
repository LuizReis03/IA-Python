# -*- coding: utf-8 -*-
"""
Created on Mon May  9 09:25:55 2022

@author: DISRCT
"""

import nltk
import matplotlib.pyplot as plt
from nltk.probability import FreqDist

#abrindo um arquivo texto
text_file = open("Artificial_Intelligence.txt")

#lendo o texto do arquivo
#e atribuindo a uma variavel
text = text_file.read()
print(text)

#separando as frases do texto
text_sentences = nltk.sent_tokenize(text)
print("Quantidade de frases:", len(text_sentences))

#separando as palavras do texto
text_words = nltk.word_tokenize(text)
print("Quantidade de palavras:", len(text_words), "\n")
print(text_words)

#verificando a frequencia das palavras
words_freq = FreqDist(text_words)
print(words_freq.most_common(10))

#vizualizando frequencia
words_freq.plot(10)

#removendo pontuação, numeração e caracteres especiais
text_words_no_punc = []

for w in text_words:
    if w.isalpha():
        text_words_no_punc.append(w.lower())
    
#comparando antes e depois
print("Quantidade de palavras antes da remoção: ", len(text_words))
print("Quantidade de palavras depois da remoção: ", len(text_words_no_punc))

print(text_words_no_punc)

#verificando a frequencia das palavras
words_no_punc_freq = FreqDist(text_words_no_punc)
print(words_no_punc_freq.most_common(10))

#vizualizando a nova frequencia
words_no_punc_freq.plot(10)

#importando a função de stopwords
from nltk.corpus import stopwords

stopwords = stopwords.words("english")
print(stopwords)

#removendo stopwords
clean_words = []

for w in text_words_no_punc:
    if w not in stopwords:
        clean_words.append(w)
        
#comparando antes e depois da remoção de stopwords
print("\nQuantidade de palavras antes: ", len(text_words_no_punc))
print("Quantidade de palavras depois: ", len(clean_words), "\n")
print(clean_words)

#importando o agoritmo de stremming
from nltk.stem import SnowballStemmer

#especificando a linguagem 
snowball_stemmer = SnowballStemmer("english")
words_stemming = []

#aplicando stemming as palavras 
for w in clean_words:
    word = snowball_stemmer.stem(w)
    words_stemming.append(word)

#verificando as palavras
print("\nTamanho:", len(words_stemming))
print(words_stemming)

#retirando palavras repetidas
words_stemming = list(set(words_stemming))
print("\nTamanho atual:", len(words_stemming))
print(words_stemming)

#Importando o algoritmo de lematização
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
words_lemma = []

#aplicando lemmatization as palavras
for w in clean_words:
    word = lemmatizer.lemmatize(w)
    words_lemma.append(word)
    
#retirando palavras repetidas
words_lemma = list(set(words_lemma))
print("\nTamanho atual:", len(words_lemma))
print(words_lemma)





        
        
        
        