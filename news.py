#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 21:01:59 2019

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
df1=df[df.real_news==1]
del df
files=df1.files.tolist()
del df1
keyList=['news']

edict={'articles':files}
exp=keyList[0]
n={exp:edict}
print(json.dumps(n, sort_keys=True,indent=2))
