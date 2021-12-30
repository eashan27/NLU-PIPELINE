from sklearn.base import BaseEstimator, TransformerMixin
import torch
from simpletransformers.classification import MultiLabelClassificationModel

class MultiIntentAnalysis(BaseEstimator, TransformerMixin):
	def __init__(self):
		device = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		if device=='cpu':
			use_cuda = False
		else:
			use_cuda = True
			
		model_args = {"use_multiprocessing": False}
		#model = MultiLabelClassificationModel('roberta', 'checkpoint-3610-epoch-5',use_cuda=False)
		self.model = MultiLabelClassificationModel('roberta', '../models/checkpoint-3115-epoch_5',use_cuda=use_cuda,args=model_args)
		self.class_name = ['atis_flight', 'atis_flight_time', 'atis_airfare', 'atis_aircraft',
       'atis_ground_service', 'atis_airport', 'atis_airline',
       'atis_distance', 'atis_abbreviation', 'atis_ground_fare',
       'atis_quantity', 'atis_city', 'atis_flight_no', 'atis_capacity',
       'atis_flight#atis_airfare', 'atis_meal', 'atis_restriction',
       'atis_airline#atis_flight_no',
       'atis_ground_service#atis_ground_fare',
       'atis_airfare#atis_flight_time', 'atis_cheapest']

	def fit(self):
		print('\n>>>>>>>fit() called. \n')
		return self


	def convert_to_dict(self,res,pred,text):
		intent=   {"value": res,
						 "confidence": pred,
						 "intent": "multi_intent",
						 "extractor": "multi_intent_extractor"}

		text['intent']=intent

		return text


	def transform(self,text):
		#print('##############################################################')
		#print(text)
		intent_res=text['correctness']['value']
		#print(intent_res)
		#print(text['query_correct_res']['query_type_result']['text'])
		#print(self.X)
		key=text['text']
		#print(key)
		prediction = self.model.predict([key])
		result = prediction[0]
		#print(result)
		probability = prediction[1]
		#print(probability)
		res = list()
		prob = list()

		for i in range(len(self.class_name)):
			if result[0][i] == 1:
				res.append(self.class_name[i])
				prob.append(probability[0][i])
		dic = dict()
		dic["result"] = res
		dic["probability"] = prob
		if intent_res=='correct':
			if len(res)!=0:
				intent =self.convert_to_dict(res,str(prob),text)
				return intent
			else:
				intent =self.convert_to_dict('out of scope',0,text)
				return intent
		else:
			intent=self.convert_to_dict('out of scope',0,text)
			return intent



