from sklearn.base import BaseEstimator, TransformerMixin
from simpletransformers.classification import ClassificationModel
import torch
import math

class QueryCorrectnessAnalysis(BaseEstimator, TransformerMixin):

	def __init__(self):
		#print(text)
		device = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		if device=='cpu':
		    use_cuda = False
		else:
		    use_cuda = True

		# if text == None:
		# 	self.X = None
		# else:
		# 	self.X = text

		# Create a ClassificationModel
		# model_args = {"use_multiprocessing": False}
  #       self.model= ClassificationModel('bert', 'query_correctness-checkpoint-38000/',use_cuda=use_cuda,args=model_args)
		self.model= ClassificationModel('bert', '../models/query_correctness-checkpoint-38000/',use_cuda=use_cuda)



	def fit(self):
		print('\n>>>>>>>fit() called. \n')
		return self



	def convert_to_dict(self,value, confidence,text):
	    """Convert model output into the Rasa NLU compatible output format."""
	    
	    query_correctness = {
	    		  "value": value,
	              "confidence": confidence,
	              "entity": "query",
	              "extractor": "query_correctness_extractor"}
	    #print(text)
	    text['correctness']=query_correctness


	    return text

	

	def transform(self,text):
		#print(text)
		#print(text['text'])
		# if self.message==None:
		# 	return "No string given"

		prediction = self.model.predict([text['text']])
		if prediction[0][0]==0:
		    result = "non-correct"
		else:
		    result = "correct"
		dic = dict()
		dic2= dict()
		dic["result"] = result
		#print(result)
		#print(prediction)
		e_non_correct = math.exp(prediction[1][0][0])/(math.exp(prediction[1][0][0])+math.exp(prediction[1][0][1]))
		e_correct= math.exp(prediction[1][0][1])/(math.exp(prediction[1][0][0])+math.exp(prediction[1][0][1]))
		#if result=='correct':
		dic["probability"] = {
		                    "non-correct": e_non_correct,
		                    "correct": e_correct
		                }
		if result=='correct':
		    query_correctness =self.convert_to_dict(result,str(dic["probability"][result]),text)
		    return query_correctness
		   
		else:
		    query_correctness =self.convert_to_dict(result,str(1-dic["probability"][result]),text)
		    return query_correctness
		   
#		message.set("correctness", [query_correctness], add_to_output=True)




