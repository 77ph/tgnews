#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:27:01 2019

@author: innerm
"""
import json
import pandas as pd

file_en='stage41.csv'
file_ru='stage42.csv'

df1=pd.read_csv(file_en)
df2=pd.read_csv(file_ru)

files_en=df1.files.tolist()
files_ru=df2.files.tolist()
del df1,df2
#keyList=['en']
#de={}
#for i in keyList: 
#    de[i] = {'Lang_code=', 'en',{'articles:', files_en[i]}}
ld=['en','ru']
edict={'articles':files_en[:10]}
exp='lang_code='+ld[0]
de={exp:edict}
print(json.dumps(de, sort_keys=True,indent=2))

edict={'articles':files_ru}
exp='lang_code='+ld[1]
dr={exp:edict}
print(json.dumps(dr, sort_keys=True,indent=2))
