from flask import Flask,render_template, request
from datetime import datetime
from flask import request,jsonify
import uuid
import time
import traceback
import configparser
import os
import logging
import json
import sys
import requests

sys.path.append("../code/")
from init_pipeline import *


config=configparser.RawConfigParser()
config.read(os.path.join(os.getcwd() , "../config/config.property"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(thread)d:%(threadName)s:%(process)d:%(message)s")
file_handler = logging.FileHandler('../logs/detectFrame.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(logging.StreamHandler())

STATUSMESSAGE = dict(config.items("STATUSMESSAGE"))
messageL20033=STATUSMESSAGE["l20033"]
messageE50017=STATUSMESSAGE["e50017"]
messageE50018=STATUSMESSAGE["e50018"]

APIENDPOINT = dict(config.items('APIENDPOINT'))
port_no = APIENDPOINT['port_no']

app=Flask(__name__)

@app.route('/home')
def Home():
    return render_template('index.html')



@app.route('/pipeline_api',methods=['POST'])
def generate_response():
	#d={'text':'what is the cheapest fare i can get from dallas to denver  '}

	start_time = time.time()
	# requests.post(json=d)
	# a=str(request.form)
	# print(a)
	# x = {'text':str(request.form['Query'])}
	# print(type(x))
	# print(x)
	requestsParam = request.get_json()
	# requestsParam=json.dumps(x)
	# print('2',requestsParam)
	# print(type(requestsParam))
	# requestsParam=json.loads(requestsParam)
	query=requestsParam['text']
	dic={'text':query}
	#print(dic)
	status,result=pipeline(dic)
	x=result
	# print(x)
	#print(x)
	# if x:
	# 	return render_template('index.html',x=x)
	# else:
	# 	return render_template('index.html','')
	if status:
		statuscode='l20033'
		return jsonify({'status':statuscode,'result':result,"statusMessage": messageL20033,"timeTaken":time.time()-start_time})
	else:
		statuscode='e50017'
		return jsonify({'status':statuscode,'result':'',"statusMessage":messageE50017,"timeTaken":time.time()-start_time})


# @app.errorhandler(Exception)

# def handle_exception(e):
#     start = time.time()
#     logger.exception("UNHANDLED EXCEPTION IN THE API")
#     return jsonify({"statusCode":"E50018",
#                     "statusMessage":messageE50018,
#                     "timeTakenForResponse":time.time()-start})

if __name__== '__main__':
	app.run(host='0.0.0.0',port=port_no,debug=True)