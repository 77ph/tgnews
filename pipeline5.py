#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 23:27:13 2019

@author: innerm
"""

import gc
import sys
import os
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import cld2
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from collections import Counter
import guidedlda
import pickle

CodeDir = os.path.dirname(os.path.realpath('pipeline6.py'))
path1 = sys.argv[1]
path=CodeDir+path1

fileout1='stage1.csv'
fileout2='stage2.csv'
fileout31='stage31.csv'
fileout32='stage32.csv'
fileout41='stage41.csv'
fileout42='stage42.csv'
files = []
for r, d, f in os.walk(path):
    for file in f:
        if '.html' in file:
            files.append(os.path.join(r, file))

url_list=[]
site_name_list=[]
title_list=[]
desc_list=[]
pubtime_list=[]
time_list=[]
text_list=[]

print(len(files),':html files detected in data directory')

files1=files

for fname in files1:
    with open(fname, "r") as f:
        contents = f.read()
        soup = BeautifulSoup(contents, 'lxml')
        text=soup.getText()
        if text:
            text=text.strip()
            text_list.append(text)
        else:
            text_list.append('')
        url = soup.find("meta",  property="og:url")
        if url:
            url_list.append(url['content'])
        else:
            url_list.append('')
        
        site_name = soup.find("meta",  property="og:site_name")
        if site_name:
            site_name_list.append(site_name['content'])
        else:
            site_name_list.append('')
        
        title = soup.find("meta",  property="og:title")
        if title:
            title_list.append(title['content'])
        else:
            title_list.append('')
        
        desc = soup.find("meta",  property="og:description")
        if desc:
            desc_list.append(desc['content'])
        else:
            desc_list.append('')
            df=pd.DataFrame()

df['files']=pd.Series(files1) 
df['url']=pd.Series(url_list) 
df['site_name']=pd.Series(site_name_list)    
df['title']=pd.Series(title_list)
df['desc']=pd.Series(desc_list) 
#df['pubtime']=pd.Series(pubtime_list)    
#df['time']=pd.Series(time_list)   ,
df['text']=pd.Series(text_list)
    
df.to_csv(fileout1,mode='w')



del df, files, files1, url_list,site_name_list, title_list,desc_list,text_list

chunksize = 50000


def lang_detect(text):
    printable_str = ''.join(x for x in text if x.isprintable())
    isReliable, textBytesFound, details = cld2.detect(printable_str)
    language_code=details[0].language_code
    if language_code != 'en' and language_code != 'ru':
        res=0
    elif language_code == 'ru':
        res=2           
    elif language_code == 'en':
        res=1          
    return res

df_ru=pd.DataFrame()
df_en=pd.DataFrame()

for df in tqdm(pd.read_csv(fileout1,chunksize=chunksize)):
    
    df['lang']=df.text.apply(lang_detect)
    df2=df[df['lang']==2]
    df1=df[df['lang']==1]
    df_ru=df_ru.append(df2)
    df_en=df_en.append(df1)

dfo=pd.DataFrame()
dfo=dfo.append(df_en)
del df_en
dfo=dfo.append(df_ru)
del df_ru
dfo.to_csv(fileout2,mode='w')

print('files dataset is:',len(dfo))

del df, dfo

print("starting stage 3 -detect themes ...")
print()


def print_top_words(model, feature_names, n_top_words):
 for topic_idx, topic in enumerate(model.components_):
     message = "Topic #%d: " % topic_idx
     message += " ".join([feature_names[i]
                          for i in topic.argsort()[:-n_top_words - 1:-1]])
     print(message)
 print()

def count_words(data):
 return len(data)

def isnews(text):
 if  'news' in text:
     res=1
 else:
     res=0
 return res

def istheme(text,theme):
 wl=[]
 for item in theme:
     wl = [ele for ele in theme if(ele in text)] 
     if wl != []:
         res=1
     else:
         res=0    
 return res

def def_news(data):
 return[isnews(text) for text in data]
 
def def_theme(data,theme):
 return[istheme(text,theme) for text in data]
 
def def_length(data):
 return[count_words(text) for text in data]
 
Society =['society','politics', 'elections', 'Legislation', 'incidents', 'crime','education','election']
#Society =['society']
Economy =['economy', 'markets', 'finance', 'business']
#Economy =['economy']
Technology =['gadgets', 'auto', 'apps', 'crypto', 'blockchain','computer','technology','iphone','android','api','software']
#Technology =['technology']
Entertainment = ['entertainment','movie', 'music', 'book', 'art','film','rock','comedy','tv-show','hbo','disney','kino','teatr','kultura']
#Entertainment = ['entertainment']
Sport=['sports','football','hockey','cricket','sport','rugby','tennis','boxing','athletics','game','rfs','dynamo','spartak','match','games']
#Sport=['sport']
Science = ['science', 'biolog', 'physics', 'genetic','math','chemistry','nauka','genom','kosmos','nano']
#Science = ['science']
categories=[(1,'society'),(2,'economy'),(3,'technology'),(4,'entertainment'),(5,'science'),(6,'sport')]

n_features = 1000
n_components = 100
n_top_words = 20
ru=pd.read_csv('stops_ru.txt',header=None)
russian=ru[0].tolist()

df=pd.read_csv(fileout2)

""" treat english part """

df_en=df[df['lang']==1]
del df
data_samples = df_en.text.values.tolist()
url=df_en.url.values.tolist()
files=df_en.files.values.tolist()

dfo=pd.DataFrame()
dfo['files']=df_en['files']
dfo['url']=df_en['url']
dfo['text']=df_en['text']
dfo['title']=df_en['title']
length=def_length(data_samples)
del df_en
dfo['length']=pd.Series(length)
dfo['news']=def_news(url)
dfo['sport']=def_theme(url,Sport)
dfo['society']=def_theme(url,Society)
dfo['economy']=def_theme(url,Economy)
dfo['technology']=def_theme(url,Technology)
dfo['entertainment']=def_theme(url,Entertainment)
dfo['science']=def_theme(url,Science)

del length
docs=dict(zip(files, data_samples))
n_samples = len(data_samples)

n_nonzero = 0
vocab=[]
for docterms in docs.values():
    unique_terms = set(docterms)    # all unique terms of this doc
    vocab.extend(unique_terms)           # set union: add unique terms of this doc
    n_nonzero += len(unique_terms)  # add count of unique terms in this doc

vocab = list(dict.fromkeys(vocab))
docnames = list(docs.keys())

docnames = np.array(docnames)
vocab = np.array(list(vocab)) 
vocab_sorter = np.argsort(vocab) 

print("done get data.")

""" create en_gLDA pipeline """
print("Extracting tf features for gLDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
tf = tf_vectorizer.fit_transform(data_samples)
gLDA = guidedlda.GuidedLDA(n_topics=n_components, n_iter=100, random_state=7, refresh=20)
gLDA.fit(tf)

filename_glda = 'glda_model.sav'


#gLDA = pickle.load(open(filename_glda, 'rb'))
#gLDA.fit(tf)
pickle.dump(gLDA, open(filename_glda, 'wb'))
## Use tf-idf features for SGD.
#print("Extracting tf-idf features for SGD...")
#tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
#                                   max_features=n_features,
#                                   stop_words='english')
#t0 = time()
#tfidf = tfidf_vectorizer.fit_transform(data_samples)
#dfo.insert(loc=0, column='tfidf', value=tfidf)
#print("done in %0.3fs." % (time() - t0))
#
#
#print()


#print("Fitting LDA models with tf features, "
#      "n_samples=%d and n_features=%d..."
#      % (n_samples, n_features))
#gLDA = guidedlda.GuidedLDA(n_topics=n_components, n_iter=100, random_state=7, refresh=20)
#gLDA.fit(tf)
#print("done in %0.3fs." % (time() - t0))

print("\nTopics in gLDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(gLDA, tf_feature_names, n_top_words)

doc_topic = gLDA.doc_topic_
#for i in range(10):
#    print("{} (top topic: {})".format([i], doc_topic[i].argmax()))
#for i in range(10):
#    print("{} (top topic: {})".format(docnames[i], doc_topic[i].argmax()))
    
print("done en LDA.")

tlist=[]

for i in range(len(dfo)):
    t2=doc_topic[i].argmax()
    tlist.append(t2)

dfo.insert(loc=0, column='t1', value=tlist)

dft=pd.DataFrame()
topic_words=[]
cat_list=[]
for item in categories:
    colname=item[1]
    colind=item[0]
    df1=dfo[dfo[colname]==1]
    
    size=len(df1)
    top=df1['t1'].tolist()
    c=Counter(top).most_common()
    most_topic=c[0][0]
    df1=df1[df1['t1']==most_topic]
    
    t_cat=(most_topic,colind)
    cat_list.append(t_cat)
    dft=dft.append(df1)
    
dft = dft.sample(frac=1).reset_index(drop=True)
msk = np.random.rand(len(dft)) < 0.7
train = dft[msk]
test = dft[~msk]
del dft
print('size of the test data:',len(test))
print("done test train data")

SFD_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=1000, tol=0.001)),
    ])
    
SFD_clf.fit(train.text, train.t1)

filename_SFD = 'SFD_model.sav'
#SFD_clf = pickle.load(open(filename_SFD, 'rb'))
#SFD_clf.fit(train.text,train.t1)
pickle.dump(SFD_clf, open(filename_SFD, 'wb'))

predicted = SFD_clf.predict(test.text)
score_test=np.mean(predicted == test.t1)
print(score_test)
theme=SFD_clf.predict(dfo.text.values)
for item in cat_list:
    theme=[item[1] if x==item[0] else x for x in theme]

dfo.insert(loc=0, column='theme', value=theme)
dfo.to_csv(fileout31,mode='w')


print("done english part ")

df=pd.read_csv(fileout2)
df_ru=df[df['lang']==2]
del df
data_samples = df_ru.text.values.tolist()
url=df_ru.url.values.tolist()
files=df_ru.files.values.tolist()

dfo=pd.DataFrame()
dfo['files']=df_ru['files']
dfo['url']=df_ru['url']
dfo['text']=df_ru['text']
dfo['title']=df_ru['title']
length=def_length(data_samples)
del df_ru
dfo['length']=pd.Series(length)
dfo['news']=def_news(url)
dfo['sport']=def_theme(url,Sport)
dfo['society']=def_theme(url,Society)
dfo['economy']=def_theme(url,Economy)
dfo['technology']=def_theme(url,Technology)
dfo['entertainment']=def_theme(url,Entertainment)
dfo['science']=def_theme(url,Science)

del length
docs=dict(zip(files, data_samples))
n_samples = len(data_samples)

n_nonzero = 0
vocab=[]
for docterms in docs.values():
    unique_terms = set(docterms)    # all unique terms of this doc
    vocab.extend(unique_terms)           # set union: add unique terms of this doc
    n_nonzero += len(unique_terms)  # add count of unique terms in this doc

vocab = list(dict.fromkeys(vocab))
docnames = list(docs.keys())

docnames = np.array(docnames)
vocab = np.array(list(vocab)) 
vocab_sorter = np.argsort(vocab) 

r=pd.read_csv('stops_ru2.txt',header=None)
russian2=r[0].tolist()

""" create en_gLDA pipeline """
tfru_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                token_pattern=r"(?u)\b\w+\b",
                                stop_words=russian2)
tfru = tfru_vectorizer.fit_transform(data_samples)
gLDAru =  guidedlda.GuidedLDA(n_topics=n_components, n_iter=100, random_state=7, refresh=20)

gLDAru.fit(tfru)

filename_gldaru = 'gldaru_model.sav'
#gLDAru = pickle.load(open(filename_gldaru, 'rb'))
#gLDAru.fit(tfru)
pickle.dump(gLDAru, open(filename_gldaru, 'wb'))

print("\nTopics in gLDAru model:")
tfru_feature_names = tfru_vectorizer.get_feature_names()
print_top_words(gLDAru, tfru_feature_names, n_top_words)
#for i in range(10):
#    print("{} (top topic: {})".format([i], doc_topic[i].argmax()))
#for i in range(10):
#    print("{} (top topic: {})".format(docnames[i], doc_topic[i].argmax()))
doc_topic = gLDAru.doc_topic_    

print('Prepare supervized data')

tlist=[]

for i in range(len(dfo)):
    t2=doc_topic[i].argmax()
    tlist.append(t2)

dfo.insert(loc=0, column='t1', value=tlist)

dft=pd.DataFrame()
topic_words=[]
cat_list=[]
for item in categories:
    colname=item[1]
    colind=item[0]
    df1=dfo[dfo[colname]==1]
    
    size=len(df1)
    top=df1['t1'].tolist()
    c=Counter(top).most_common()
    most_topic=c[0][0]
    df1=df1[df1['t1']==most_topic]
    
    t_cat=(most_topic,colind)
    cat_list.append(t_cat)
    dft=dft.append(df1)
    
dft = dft.sample(frac=1).reset_index(drop=True)
msk = np.random.rand(len(dft)) < 0.7
train = dft[msk]
test = dft[~msk]
del dft

print("done test train ru")

SFD_clfru = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=1000, tol=0.001)),
    ])
    
SFD_clfru.fit(train.text, train.t1)

filename_SFDru = 'SFDru_model.sav'
#SFD_clfru = pickle.load(open(filename_SFDru, 'rb'))
#SFD_clfru.fit(train.text, train.t1)
pickle.dump(SFD_clfru, open(filename_SFDru, 'wb'))

predicted = SFD_clfru.predict(test.text)
score_test=np.mean(predicted == test.t1)
print(score_test)
theme=SFD_clfru.predict(dfo.text.values)
for item in cat_list:
    theme=[item[1] if x==item[0] else x for x in theme]

dfo.insert(loc=0, column='theme', value=theme)
dfo.to_csv(fileout32,mode='w')
del dfo
print("done train ru data")

""" end russian part """
gc.collect()
""" end stage3 """
print("end stage 3")

""" stage 4 """

n_features = 1000
n_top_words = 20
ru=pd.read_csv('stops_ru.txt',header=None)
russian=ru[0].tolist()

fileout31='stage31.csv'
fileout32='stage32.csv'

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
def rate_thread(topic,c):
    for item in c:
        if item[0] == topic:
            res=item[1]
    return res

def set_news(rate_thread,size,item):
    if item!=1:
        a=rate_thread/size
        if a>0.001:
            res = 1
        else:
            res = 0
    else:
        res = 1
    return res


"""  english files treat"""
print("treat english part")



dfo=pd.DataFrame()

for i in range(1,7):

    df=pd.read_csv('stage31.csv')
    dft=pd.DataFrame()
    df1=df[df['theme']==i]
    del df
    dft['files']=df1['files']
    dft['length']=df1['length']
    dft['text']=df1['text']
    dft['title']=df1['title']
    dft['theme']=df1['theme']
    dft['news']=df1['news']
    del df1
    data_samples = dft.text.tolist()
    n_samples = len(data_samples)
    n_components = 200
    files=dft.files.tolist()

    print("Extracting tf features for gLDA...")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
    tf = tf_vectorizer.fit_transform(data_samples)

    filename_glda2 = 'glda_model2.sav'
    print("train LDA with new data")
    gLDA2 = guidedlda.GuidedLDA(n_topics=n_components, n_iter=100, random_state=7, refresh=20)
    gLDA2.fit(tf)

#gLDA2 = pickle.load(open(filename_glda2, 'rb'))
#gLDA2.fit(tf)
#    pickle.dump(gLDA2, open(filename_glda2, 'wb'))

    print("\nTopics in gLDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(gLDA2, tf_feature_names, n_top_words)

    doc_topic = gLDA2.doc_topic_
    
    tlist=[]

    for i in range(len(dft)):
        t2=doc_topic[i].argmax()
        tlist.append(t2)
        c=Counter(tlist).most_common()
    dft.insert(loc=0, column='thread', value=tlist)

    plist=[]

    for i in range(len(dft)):
        p1=doc_topic[i].max()
        plist.append(p1)
    dft.insert(loc=0, column='prob', value=plist)



    dft['rate_thread']=dft.apply(lambda row: rate_thread(row['thread'],c),axis=1)
    dft['real_news']=dft.apply(lambda row: set_news(row['rate_thread'],len(tlist),row['news']),axis=1)
    dft['theme']=dft.apply(lambda row: 7 if row['prob']  < 0.12 else row['theme'],axis=1)
    dfo=dfo.append(dft)
    
dfo.to_csv('stage41.csv',mode='w')
del dfo

print("treat russian part")
n_features = 1000
n_top_words = 5

r=pd.read_csv('stops_ru2.txt',header=None)
russian2=r[0].tolist()
stop_words=[]
dfo=pd.DataFrame()

for i in range(1,7):

    df=pd.read_csv('stage32.csv')
    dft=pd.DataFrame()
    df1=df[df['theme']==i]
#    del df
    dft['files']=df1['files']
    dft['length']=df1['length']
    dft['text']=df1['text']
    dft['title']=df1['title']
    dft['theme']=df1['theme']
    dft['news']=df1['news']
#    del df1
    data_samples = dft.text.tolist()
    n_samples = len(data_samples)
    n_components = 200
    files=dft.files.tolist()

    print("Extracting tf features for gLDA...")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                decode_error='ignore',
                                max_features=n_features,
                                stop_words=russian2,
                                )
    
    try:
        tf = tf_vectorizer.fit_transform(data_samples)
    

    #filename_glda2 = 'glda_model2.sav'
        print("train LDA with new data")
        gLDA2 = guidedlda.GuidedLDA(n_topics=n_components, n_iter=100, random_state=7, refresh=20)
        gLDA2.fit(tf)

    #gLDA2 = pickle.load(open(filename_glda2, 'rb'))
    #gLDA2.fit(tf)
#    pickle.dump(gLDA2, open(filename_glda2, 'wb'))

#    print("\nTopics in gLDA model:")
        tf_feature_names = tf_vectorizer.get_feature_names()
        print_top_words(gLDA2, tf_feature_names, n_top_words)

        doc_topic = gLDA2.doc_topic_

#    for i in range(10):
#        print("{} (top topic: {})".format([i], doc_topic[i].argmax()))
   

        tlist=[]

        for i in range(len(dft)):
            t2=doc_topic[i].argmax()
            tlist.append(t2)
            c=Counter(tlist).most_common()
        dft.insert(loc=0, column='thread', value=tlist)

        plist=[]

        for i in range(len(dft)):
            p1=doc_topic[i].max()
            plist.append(p1)
        dft.insert(loc=0, column='prob', value=plist)



        dft['rate_thread']=dft.apply(lambda row: rate_thread(row['thread'],c),axis=1)
        dft['real_news']=dft.apply(lambda row: set_news(row['rate_thread'],len(tlist),row['news']),axis=1)
        dft['theme']=dft.apply(lambda row: 7 if row['prob']  < 0.12 else row['theme'],axis=1)
        dfo=dfo.append(dft)
    except:
        print('russian traing dataset not enough for theme:', categories[i-1])
dfo.to_csv('stage42.csv',mode='w')
del dfo





