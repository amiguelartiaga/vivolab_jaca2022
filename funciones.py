import numpy as np
import requests
import justext as jt


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
# from sklearn.linear_model import LogisticRegression

def entrenar_clasificador1(train, labels):
    vec = CountVectorizer(max_features= 1000, preprocessor=limpiar_texto)    
    x = vec.fit_transform(train)
    # print(vec.get_feature_names_out())
    # x = x.toarray()
    model = MultinomialNB().fit(x, labels)
    # model = LogisticRegression().fit(x.astype(np.float32), labels)
    return model, vec

def evaluar_clasificador1(model, vec, texto):
    x = vec.transform([texto])
    # x = x.toarray().astype(np.float32)
    return model.predict(x)

def entrenar_clasificador2(train, labels):
    vec = CountVectorizer(max_features= 1000, preprocessor=limpiar_texto)    
    x = vec.fit_transform(train)
    # print(vec.get_feature_names_out())  
    tfidf = TfidfTransformer()
    x = tfidf.fit_transform(x)
    # x = x.toarray()
    # y = np.array(labels)
    model = MultinomialNB().fit(x, labels)
    # model = LogisticRegression().fit(x.astype(np.float32), y)
    return model, vec, tfidf
    
def evaluar_clasificador2(model, vec, tfidf, texto):
    x = vec.transform([texto])
    x = tfidf.transform(x)
    # x = x.toarray().astype(np.float32)
    return model.predict(x)

