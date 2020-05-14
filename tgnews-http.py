#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 14 May 2020 09:02:03 AM EDT

@author: al

Requires:
Debian 10 x64
sudo apt update && sudo apt -y upgrade
pip3 install eve
pip3 install greenlet

Example:
chmod 755 tgnews-http
./tgnews-http.py 5000

Request:
apt-get install curl
2.3. 
curl -L -k -v "http://127.0.0.1:5000/threads?period=111&lang_code=ru&category=11"
2.1.
curl -v -X PUT -T "20200427/00/1003461414779765232.html" "http://127.0.0.1:5000/1003461414779765232.html"
2.2.
curl -v -X DELETE "http://127.0.0.1:5000/1003461414779765232.html"
"""

import sys
import os
from flask import Flask, Response, request
import logging

def insert(filename,body):
    print("DEBUG call insert :: file::",filename)
    print("DEBUG call insert :: body::",body)
    res = "Created"
    return res

def delete(filename):
    print("DEBUG call delete :: delete file::",filename)
    res = "Deleted"
    return res


app = Flask(__name__)
# file_handler = logging.FileHandler('app.log')
# app.logger.addHandler(file_handler)
# app.logger.setLevel(logging.INFO)


PORT = 5000
if sys.argv[1:]:
   PORT = sys.argv[1]

if int(PORT) < 1024:
   if os.getuid() != 0:
       print("port = {0} Bind ports below 1024 need root capabilites on Linux".format(PORT))
       exit(1)

ready = True

# https://stackoverflow.com/questions/15974730/how-do-i-get-the-different-parts-of-a-flask-requests-url
@app.route('/<rr>',methods = ['GET', 'PUT', 'DELETE'])
def api_root(rr):
    if ready:
        if request.method == 'GET':
            ### stub -- GET /threads?period=<period>&lang_code=<lang_code>&category=<category> HTTP/1.1
            print(request.args)
            return "OK\n"

        if request.method == 'PUT':
            filename = str(request.url_rule)
            body = request.data.decode('utf-8')
            print("DEBUG req::",request.method)
            print("DEBUG file::",filename)
            print("DEBUG body::",body)
            res = insert(filename,body)
            if res == "Created":
                #HTTP/1.1 201 Created
                text = "Created"
                resp = Response(text, status=201, mimetype='text/plain')
                return resp
            elif res == "Updated":
                text = "Updated"
                resp = Response(text, status=204, mimetype='text/plain')
                return resp
            else:
                text = "Service Unavailable"
                resp = Response(text, status=503, mimetype='text/plain')
                return resp

        if request.method == 'DELETE':
            filename = str(request.url_rule)
            body = request.data.decode('utf-8')
            print("DEBUG req::",request.method)
            print("DEBUG file::",filename)
            res = delete(filename)           
            if res == "Deleted":
                #HTTP/1.1 201 Created
                text = "No Content"
                resp = Response(text, status=204, mimetype='text/plain') 
                return resp
            elif res == "NotFound":
                text = "Not found"
                resp = Response(text, status=404, mimetype='text/plain')
                return resp
            else:
                text = "Service Unavailable"
                resp = Response(text, status=503, mimetype='text/plain')
                return resp
    
    else:
        text = "Service Unavailable"
        resp = Response(text, status=503, mimetype='text/plain')
        return resp



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=PORT)
