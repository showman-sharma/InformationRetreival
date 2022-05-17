from util import *

# Add your import statements here
from nltk.corpus import stopwords
#import nltk
#nltk.download('stopwords')



class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = []
		stop_words = set(stopwords.words("english")+ ['.',',',')','(']) 
		for tokens_list in text:
			l = []
			for word in tokens_list:
				if not word in stop_words:
					l.append(word)
			stopwordRemovedText.append(l)

		#Fill in code here

		return stopwordRemovedText




	