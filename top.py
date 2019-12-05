#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 22:26:54 2019

@author: innerm
"""
import json
import pandas as pd
import re

def clear_title(data):
    data = re.sub('\S*@\S*\s?',' ', data)
    data = re.sub('\s+', ' ',  data)
    data = re.sub("\'", " ", data)
    return data

file_en='stage41.csv'
file_ru='stage42.csv'

df=pd.DataFrame()
df1=pd.read_csv(file_en)
df2=pd.read_csv(file_ru)
df=df.append(df1)
df=df.append(df1)
del df1,df2

df=df[df.real_news==1]
categories=[(1,'society'),(2,'economy'),(3,'technology'),(4,'entertainment'),(5,'science'),(6,'sport'),(7,'others')]

df=df.sort_values(by=['rate_thread'],ascending=False)
df=df[df['rate_thread']>3]
th=df.thread.tolist()
dt = {i:th.count(i) for i in th}


for item in dt.keys():
    
    dd=df[df['thread']==item]
    dd=dd.sort_values(by=['prob'],ascending=False)
    tt=dd.title.tolist()[0]
    tt=clear_title(tt)
    files=dd.files.tolist()
    edict={'articles':files}
    exp2='thread:'+tt
    n={exp2:edict}
    print(json.dumps(n, sort_keys=True,indent=3))

for item in categories:
    df1=df[df.theme==item[0]]
    df1=df1.sort_values(by=['rate_thread'],ascending=False)
    df1=df1[df1['rate_thread']>3]
    th=df1.thread.tolist()
    dt = {i:th.count(i) for i in th}
    exp='category:'+item[1]
    for item2 in dt.keys():
        dd=df1[df1['thread']==item2]
        dd=dd.sort_values(by=['prob'],ascending=False)
        tt=dd.title.tolist()[0]
        tt=clear_title(tt)
        files=dd.files.tolist()
        edict={'articles':files}
        exp2='thread:'+tt
        n={exp2:edict}
        m={exp:n}
        print(json.dumps(m, sort_keys=True,indent=4))
