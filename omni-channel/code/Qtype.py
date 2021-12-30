from sklearn.base import BaseEstimator, TransformerMixin
from simpletransformers.classification import ClassificationModel
import torch
import math

class QueryTypeAnalysis(BaseEstimator, TransformerMixin):

	def __init__(self):
		device = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		if device=='cpu':
			use_cuda = False
		else:
			use_cuda = True

		# Create a ClassificationModel
		model_args = {"use_multiprocessing": False}
		self.model= ClassificationModel('bert', '../models/query_type-checkpoint-8000/',use_cuda=use_cuda,args=model_args)
		self.class_name = ['ENTITY','DESCRIPTION','LOCATION','NUMERIC','PERSON']



	def fit(self):
		print('\n>>>>>>>fit() called. \n')
		return self


	def convert_to_dict(self,value, confidence,text):
		"""Convert model output into the Rasa NLU compatible output format."""
		#print(text['value'])
		query_type = {"value": value,
						  "confidence": confidence,
						  "entity": "query",
						  "extractor": "query_type_extractor"}

		text['type']=query_type

		return text



	def transform(self,text):
		#print(text)
		type_res=text['correctness']['value']

		#print(text)
		prediction = self.model.predict([text['text']])
		result = self.class_name[prediction[0][0]]

		e_entity=math.exp(prediction[1][0][0])/(math.exp(prediction[1][0][0])+math.exp(prediction[1][0][1])+math.exp(prediction[1][0][2])+math.exp(prediction[1][0][3])+math.exp(prediction[1][0][4]))
		e_description=math.exp(prediction[1][0][1])/(math.exp(prediction[1][0][0])+math.exp(prediction[1][0][1])+math.exp(prediction[1][0][2])+math.exp(prediction[1][0][3])+math.exp(prediction[1][0][4]))
		e_location=math.exp(prediction[1][0][2])/(math.exp(prediction[1][0][0])+math.exp(prediction[1][0][1])+math.exp(prediction[1][0][2])+math.exp(prediction[1][0][3])+math.exp(prediction[1][0][4]))
		e_numeric=math.exp(prediction[1][0][3])/(math.exp(prediction[1][0][0])+math.exp(prediction[1][0][1])+math.exp(prediction[1][0][2])+math.exp(prediction[1][0][3])+math.exp(prediction[1][0][4]))
		e_person=math.exp(prediction[1][0][4])/(math.exp(prediction[1][0][0])+math.exp(prediction[1][0][1])+math.exp(prediction[1][0][2])+math.exp(prediction[1][0][3])+math.exp(prediction[1][0][4]))

		dic = dict()
		dic["result"] = result
		dic["confidence"] = {
							"ENTITY": e_entity,
							"DESCRIPTION":e_description,
							"LOCATION":e_location,
							"NUMERIC":e_numeric,
							"PERSON":e_person
						}
		if type_res=='correct':
			query_type =self.convert_to_dict(result,str(dic["confidence"][result]),text)
			return query_type
		else:
			query_type =self.convert_to_dict('-',0,text)
			return query_type

		# self.X = text

		# prediction = self.model.predict(self.X)
		# #print('7.',prediction)
		# result = self.class_name[prediction[0][0]]
		# #label=message.get('correctness')[0].get('value')
		# #print(label)
		# #if label=='non-correct':

		# #if label=='non-correct':
		# e_entity=math.exp(prediction[1][0][0])/(math.exp(prediction[1][0][0])+math.exp(prediction[1][0][1])+math.exp(prediction[1][0][2])+math.exp(prediction[1][0][3])+math.exp(prediction[1][0][4]))
		# e_description=math.exp(prediction[1][0][1])/(math.exp(prediction[1][0][0])+math.exp(prediction[1][0][1])+math.exp(prediction[1][0][2])+math.exp(prediction[1][0][3])+math.exp(prediction[1][0][4]))
		# e_location=math.exp(prediction[1][0][2])/(math.exp(prediction[1][0][0])+math.exp(prediction[1][0][1])+math.exp(prediction[1][0][2])+math.exp(prediction[1][0][3])+math.exp(prediction[1][0][4]))
		# e_numeric=math.exp(prediction[1][0][3])/(math.exp(prediction[1][0][0])+math.exp(prediction[1][0][1])+math.exp(prediction[1][0][2])+math.exp(prediction[1][0][3])+math.exp(prediction[1][0][4]))
		# e_person=math.exp(prediction[1][0][4])/(math.exp(prediction[1][0][0])+math.exp(prediction[1][0][1])+math.exp(prediction[1][0][2])+math.exp(prediction[1][0][3])+math.exp(prediction[1][0][4]))

		# dic = dict()
		# dic["result"] = result
		# dic["confidence"] = {
		#                     "ENTITY": e_entity,
		#                     "DESCRIPTION":e_description,
		#                     "LOCATION":e_location,
		#                     "NUMERIC":e_numeric,
		#                     "PERSON":e_person
		#                 }
		# #print('9. type_confidence',dic['confidence'])
		# #print(prediction)
		# #if label=='correct':
		# query_type =self.convert_to_dict(result,str(dic["confidence"][result]))
		# return query_type
			#message.set("type", [query_type], add_to_output=True)
		#else:
			#query_type =self.convert_to_rasa('-','0')
			#message.set("type",[query_type],add_to_output=True)

