from Qcorrectness import QueryCorrectnessAnalysis
from Qtype import QueryTypeAnalysis
from Multi_intent import MultiIntentAnalysis
# from Query_ans import QueryAnsAnalysis
from Slots import CustomSlotAnalysis
from Sentiment import SentimentAnalysis
#from Sentiment_1 import SentimentAnalysis
#from word_imp import word_importance
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

#d={'text':'who are you'}
#Dear sir/Madam Kindly pls check the my bank statement so why 400/-bounce change....every month deduct on dated 5th..so please refund my 400/-

d={'text':'i want to fly from baltimore to dallas round trip'}
# d={'text':'hau bchuu shuxddhdx dshxhh'}

def pipeline(x):
	try:
	
		print(  '###################################################',
		   		'#        CREATING PREDICTION PIPELINE             #',
			   '####################################################')
		pipe = Pipeline(steps=[
								('Query_Correctness', QueryCorrectnessAnalysis()),
								('Query_Type', QueryTypeAnalysis()),
								('Multi Intent',MultiIntentAnalysis()),
								('Slots',CustomSlotAnalysis()),
								('Sentiment',SentimentAnalysis()),
								# ('Word_importance',word_importance())
								

			])
		
		result=pipe.transform(x)

		print(result)
		return True,result

	except Exception as e:
		print(e)
		return False,''
pipeline(d)
