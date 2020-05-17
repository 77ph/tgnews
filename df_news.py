#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 12:53:29 2019

@author: innerm


sudo apt update && sudo apt -y upgrade
sudo apt install mariadb-server
sudo systemctl status mysql
sudo mysqladmin -u root version

sudo mysql -u root
MariaDB [(none)]> CREATE USER 'tgnews'@'localhost' IDENTIFIED BY '123456';       
Query OK, 0 rows affected (0.761 sec)

MariaDB [(none)]> GRANT ALL PRIVILEGES ON *.* TO 'tgnews'@'localhost' WITH GRANT OPTION;     
Query OK, 0 rows affected (0.400 sec)

MariaDB [(none)]> exit

mysql -u tgnews -p"123456"
MariaDB [(none)]> create database tgnews;
Query OK, 1 row affected (0.463 sec)

MariaDB [(none)]> exit

https://github.com/PyMySQL/PyMySQL
python3 -m pip install PyMySQL


"""

import os
#import cld2
from bs4 import BeautifulSoup
import pandas as pd
#from shutil import copyfile
#import mysql.connector as mysql

import sqlalchemy
import time
import numpy as np

# path = '/home/innerm/ML/data2'
path = './20200503'
fileout='stage1.csv'
files = []
# r=root, d=directories, f = files
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
    
        pubtime = soup.find("meta",  property="article:published_time")
        if pubtime:
            pubtime_list.append(pubtime['content'])
        else:
            pubtime_list.append(0)
    
        time = soup.find("time")
        if time:
            time_list.append(time['datetime'])
        else:
            time_list.append(0)

#df=pd.read_csv('files_by_lang-cld2.csv')
#df_en=df[df['lang']==1]
df=pd.DataFrame()

df['files']=pd.Series(files1) 
df['url']=pd.Series(url_list) 
df['site_name']=pd.Series(site_name_list)    
df['title']=pd.Series(title_list)
df['desc']=pd.Series(desc_list) 
df['pubtime']=pd.Series(pubtime_list)    
df['time']=pd.Series(time_list)   
df['text']=pd.Series(text_list)
    
df.to_csv(fileout,mode='w')

database_username = 'tgnews'
database_password = '123456'
database_ip       = '127.0.0.1'
database_name     = 'tgnews'

engine = sqlalchemy.create_engine('mysql+pymysql://{0}:{1}@{2}/{3}'.format(database_username, database_password, database_ip, database_name),pool_recycle=1)

# [SQL: INSERT INTO table_name_for_df (`index`, files, url, site_name, title, `desc`, pubtime, time, text) VALUES (%(index)s, %(files)s, %(url)s, %(site_name)s, %(title)s, %(desc)s, %(pubtime)s, %(time)s, %(text)s)]
# 'pubtime': '2020-05-03T05:20:00+00:00', 'time': '2020-05-03T05:20:00+00:00'
df.to_sql(con=engine, name='table_name_for_df', if_exists='replace', chunksize=20000, 
dtype={'files': sqlalchemy.types.NVARCHAR(length=255), 'url': sqlalchemy.types.NVARCHAR(length=4096),'site_name': sqlalchemy.types.NVARCHAR(length=255),'title': sqlalchemy.types.Text,
'desc': sqlalchemy.types.Text, pubtime: sqlalchemy.types.Text, 'time': sqlalchemy.types.Text, 'text': sqlalchemy.types.Text(length=4294000000)})

print("Export: OK")
