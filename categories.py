#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 21:16:37 2019

@author: innerm
"""

import json
import pandas as pd

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

for item in categories:
    dfc=df[df.theme==item[0]]
    files=dfc.files.tolist()
    edict={'articles':files}
    exp='category:'+item[1]
    n={exp:edict}
    print(json.dumps(n, sort_keys=True,indent=3))

    

























