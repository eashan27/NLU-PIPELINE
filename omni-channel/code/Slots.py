from sklearn.base import BaseEstimator, TransformerMixin
from simpletransformers.classification import ClassificationModel
import torch
from simpletransformers.ner import NERModel

class CustomSlotAnalysis(BaseEstimator, TransformerMixin):
	def __init__(self):
		device = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		if device=='cpu':
			use_cuda = False
		else:
			use_cuda = True
		model_args = {"use_multiprocessing": False}
		self.labels = ['B-aircraft_code',
				'B-airline_code',
				'B-airline_name',
				'B-airport_code',
				'B-airport_name',
				'B-arrive_date.date_relative',
				'B-arrive_date.day_name',
				'B-arrive_date.day_number',
				'B-arrive_date.month_name',
				'B-arrive_date.today_relative',
				'B-arrive_time.end_time',
				'B-arrive_time.period_mod',
				'B-arrive_time.period_of_day',
				'B-arrive_time.start_time',
				'B-arrive_time.time',
				'B-arrive_time.time_relative',
				'B-booking_class',
				'B-city_name',
				'B-class_type',
				'B-compartment',
				'B-connect',
				'B-cost_relative',
				'B-day_name',
				'B-day_number',
				'B-days_code',
				'B-depart_date.date_relative',
				'B-depart_date.day_name',
				'B-depart_date.day_number',
				'B-depart_date.month_name',
				'B-depart_date.today_relative',
				'B-depart_date.year',
				'B-depart_time.end_time',
				'B-depart_time.period_mod',
				'B-depart_time.period_of_day',
				'B-depart_time.start_time',
				'B-depart_time.time',
				'B-depart_time.time_relative',
				'B-economy',
				'B-fare_amount',
				'B-fare_basis_code',
				'B-flight',
				'B-flight_days',
				'B-flight_mod',
				'B-flight_number',
				'B-flight_stop',
				'B-flight_time',
				'B-fromloc.airport_code',
				'B-fromloc.airport_name',
				'B-fromloc.city_name',
				'B-fromloc.state_code',
				'B-fromloc.state_name',
				'B-meal',
				'B-meal_code',
				'B-meal_description',
				'B-mod',
				'B-month_name',
				'B-or',
				'B-period_of_day',
				'B-restriction_code',
				'B-return_date.date_relative',
				'B-return_date.day_name',
				'B-return_date.day_number',
				'B-return_date.month_name',
				'B-return_date.today_relative',
				'B-return_time.period_mod',
				'B-return_time.period_of_day',
				'B-round_trip',
				'B-state_code',
				'B-state_name',
				'B-stoploc.airport_code',
				'B-stoploc.airport_name',
				'B-stoploc.city_name',
				'B-stoploc.state_code',
				'B-time',
				'B-time_relative',
				'B-today_relative',
				'B-toloc.airport_code',
				'B-toloc.airport_name',
				'B-toloc.city_name',
				'B-toloc.country_name',
				'B-toloc.state_code',
				'B-toloc.state_name',
				'B-transport_type',
				'I-airline_name',
				'I-airport_name',
				'I-arrive_date.date_relative',
				'I-arrive_date.day_number',
				'I-arrive_time.end_time',
				'I-arrive_time.period_of_day',
				'I-arrive_time.start_time',
				'I-arrive_time.time',
				'I-arrive_time.time_relative',
				'I-city_name',
				'I-class_type',
				'I-cost_relative',
				'I-depart_date.date_relative',
				'I-depart_date.day_number',
				'I-depart_date.today_relative',
				'I-depart_time.end_time',
				'I-depart_time.period_of_day',
				'I-depart_time.start_time',
				'I-depart_time.time',
				'I-depart_time.time_relative',
				'I-economy',
				'I-fare_amount',
				'I-fare_basis_code',
				'I-flight_mod',
				'I-flight_number',
				'I-flight_stop',
				'I-flight_time',
				'I-fromloc.airport_name',
				'I-fromloc.city_name',
				'I-fromloc.state_name',
				'I-meal_code',
				'I-meal_description',
				'I-period_of_day',
				'I-restriction_code',
				'I-return_date.date_relative',
				'I-return_date.day_number',
				'I-return_date.today_relative',
				'I-round_trip',
				'I-state_name',
				'I-stoploc.city_name',
				'I-time',
				'I-today_relative',
				'I-toloc.airport_name',
				'I-toloc.city_name',
				'I-toloc.state_name',
				'I-transport_type',
				'O']
		self.model = NERModel('bert', '../models/checkpoint-1120-epoch-2',use_cuda=use_cuda,labels=self.labels,args=model_args)



	def fit(self):
		print('\n>>>>>>>fit() called. \n')
		return self



	def convert_to_dict(self,flag,ent,text):
	    l=[]
	    for i in range(len(flag)):
	        l.append({"slot":flag[i],
	                "value":ent[i],
	              "extractor": "slot_extractor"})
	    #x={"entity":l}
	    #print(l)
	    text['entities']=l

	    

	    return text



	def transform(self,text):
		slot_res=text['correctness']['value']
		#print(text['intent']['value'][0])
		#print(slot_res)
		prediction = self.model.predict([text['text']])
		#print('15:slot_pred',prediction)
		#label=message.get('correctness')[0].get('value')
		prediction = prediction[0][0]
		# print(prediction)
		#text = ''
		ent=[]
		flag=[]
		# ent=''
		# flag=''
		for i in prediction:
		    key = list(i.keys())[0]
		    if i[key]!='O':
		        flag.append(i[key])
		        ent.append(key.split('{')[0])

		if slot_res=='correct':
			entities =self.convert_to_dict(flag,ent,text)
			return entities
		else:
			entities =self.convert_to_dict('-','-',text)
			return entities