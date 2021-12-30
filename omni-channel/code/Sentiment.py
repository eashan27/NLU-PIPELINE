from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import torch
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import transformers
from transformers import pipeline
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SentimentAnalysis(BaseEstimator, TransformerMixin):
	def __init__(self):
		device = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		if device=='cpu':
			use_cuda = False
		else:
			use_cuda = True
			
		model_args = {"use_multiprocessing": False}
		self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
		self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")




	def fit(self):
		print('\n>>>>>>>fit() called. \n')
		return self


	def convert_to_dict(self, val, c,text):
		"""Convert model output into the Rasa NLU compatible output format."""
		
		sentiment = {"value": val,
				  "confidence": c,
				  "entity": "sentiment",
				  "extractor": "sentiment_extractor"}
		text['sentiment']=sentiment

		return text


	def transform(self,text):
		#print(text)
		sent_res=text['correctness']['value']
		classifier = transformers.pipeline('sentiment-analysis',model=self.model,tokenizer=self.tokenizer)
		result=classifier(text['text'])
		for i in result:
		    if i['score']>0.5:
		        key=i['label']
		        val=i['score']
		    #print(key,val)
		# for i in result:
		#     for j in i:
		#         if j['score']>0.5:
		#             key=j['label']
		#             val=j['score']
		    #print(key,val)
			
		# sid = SentimentIntensityAnalyzer()
		# res = sid.polarity_scores(text['text'])
		# if res['compound']>=0.5:
		# 	#print('Pos',res['pos'])
		# 	key='Pos'
		# 	val=res['pos']
		# elif (res['compound']>-0.5) and (res['compound']<0.5):
		# 	#print('Neu',res['neu'])
		# 	key='Neu'
		# 	val=res['neu']
		# else:
		# 	#print('Neg',res['neg'])
		# 	key='Neg'
		# 	val=res['neg']
		if sent_res=='correct':
			sentiment = self.convert_to_dict(key, val,text)
			return sentiment
		else:
			sentiment=self.convert_to_dict('-',0,text)
			return sentiment
