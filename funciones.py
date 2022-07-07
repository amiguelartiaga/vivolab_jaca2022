import numpy as np
import requests
import justext as jt
import re

def imprimir_comienzo(texto):
    texto = re.sub("\n",' ', texto)
    x = [x for x in texto.split('.')[:3]]
    txt = '. '.join(x) + '.'
    txt = re.sub(' +', ' ', txt)
    print(txt)        

def limpiar_texto_web(text):
    stop = jt.get_stoplist("Spanish")
    out = [x.text for x in jt.justext(text,stop) if not x.is_boilerplate]
    return "\n".join(out)

def descargar_web(web):
    return requests.get(web).text

def descargar_lista_webs(webs):
    textos = [descargar_web(web) for web in webs]
    textos = [limpiar_texto_web(texto) for texto in textos]
    return textos


import unicodedata
def eliminar_acentos(x):    
    try:
        x = unicode(x, 'utf-8')
    except (TypeError, NameError): # unicode is a default on python 3
        pass
    x = unicodedata.normalize('NFD', x)
    x = x.encode('ascii', 'ignore')
    x = x.decode("utf-8")
    return x

import re
from nltk.corpus import stopwords
def limpiar_texto(txt):
    txt = eliminar_acentos(txt)
    txt = txt.lower()
    txt = re.sub("[^a-z]",' ', txt)
    txt = [ w for w in txt.split(' ') if w not in stopwords.words('spanish') ]
    txt = [ w for w in txt if len(w) > 1]
    txt = " ".join(txt)
    txt = re.sub(' +', ' ', txt)
    return txt

import pandas as pd
def ver_datos(x, cols, y):
    df = pd.DataFrame(x, columns=cols)
    df['etiqueta'] = y
    print( df ) 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

def entrenar_clasificador1(train, labels):
    vec = CountVectorizer(max_features= 1000, preprocessor=limpiar_texto)    
    x = vec.fit_transform(train)
    model = MultinomialNB().fit(x, labels)
    return model, vec

def evaluar_clasificador1(model, vec, texto):
    x = vec.transform([texto])
    return model.predict(x)

def entrenar_clasificador2(train, labels):
    vec = CountVectorizer(max_features= 1000, preprocessor=limpiar_texto)    
    x = vec.fit_transform(train)
    tfidf = TfidfTransformer()
    x = tfidf.fit_transform(x)
    model = MultinomialNB().fit(x, labels)
    return model, vec, tfidf
    
def evaluar_clasificador2(model, vec, tfidf, texto):
    x = vec.transform([texto])
    x = tfidf.transform(x)
    return model.predict(x)

def word2vec_linea(word2vec, line):
    x = [word2vec[x] for x in line.split() if x in word2vec] 
    return sum(x)/len(x)

def entrenar_clasificador3(word2vec, train, labels):
    train = [ limpiar_texto(x) for x in train ]
    train = [ word2vec_linea(word2vec,linea) for linea in train]
    x = np.array(train)
    model = LogisticRegression().fit(x, labels)
    return model

def evaluar_clasificador3(word2vec, model, texto):
    train = [ limpiar_texto(texto) ]
    train = [ word2vec_linea(word2vec,linea) for linea in train]
    x = np.array(train)
    return model.predict(x)


from sklearn.cluster import KMeans
def clustering1(word2vec, textos, numero_clusters):
    textos = [ limpiar_texto(x) for x in textos ]
    textos = [ word2vec_linea(word2vec,linea) for linea in textos]
    x = np.array(textos)
    kmeans = KMeans(n_clusters=numero_clusters, random_state=0).fit(x)
    return kmeans.labels_

from sklearn.cluster import SpectralClustering
def clustering2(word2vec, textos, numero_clusters):
    textos = [ limpiar_texto(x) for x in textos ]
    textos = [ word2vec_linea(word2vec,linea) for linea in textos]
    x = np.array(textos)
    cl = SpectralClustering(assign_labels='discretize', n_clusters=numero_clusters,random_state=0).fit(x)
    return cl.labels_


from sklearn.metrics.pairwise import cosine_similarity
def preparar_buscador1(train):
    vec = CountVectorizer(max_features= 1000, preprocessor=limpiar_texto)    
    x = vec.fit_transform(train)
    tfidf = TfidfTransformer()
    x = tfidf.fit_transform(x)
    return vec, tfidf, x

def consulta_buscador1(vec, tfidf, xdb, textosdb, texto, topk=3):
    x = vec.transform([texto])
    x = tfidf.transform(x)
    s = cosine_similarity(xdb, x).reshape(-1)
    ind = s.argsort()[-topk:][::-1]
    for n, i in enumerate(ind):
        print("\n(%d/%d) %f: " % (n+1, len(ind), s[i]))
        imprimir_comienzo(textosdb[i])
        
def preparar_buscador2(word2vec, train):
    train = [ limpiar_texto(x) for x in train ]
    train = [ word2vec_linea(word2vec,linea) for linea in train]
    x = np.array(train)    
    return x

def consulta_buscador2(word2vec, xdb, textosdb, texto, topk=3):
    texto = [ limpiar_texto(texto) ]
    texto = [ word2vec_linea(word2vec,linea) for linea in texto]
    x = np.array(texto)
   
    s = cosine_similarity(xdb, x).reshape(-1)
    ind = s.argsort()[-topk:][::-1]
    for n, i in enumerate(ind):
        print("\n(%d/%d) %f: " % (n+1, len(ind), s[i]))
        imprimir_comienzo(textosdb[i])
        
        