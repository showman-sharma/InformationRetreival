from util import *

# Add your import statements here
import re
from nltk.tokenize.punkt import PunktSentenceTokenizer


class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = re.split('\? |\! |\. |\n',text)

		#Fill in code here


		return segmentedText





	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		segmentedText = None

		#Fill in code here
		pst = PunktSentenceTokenizer()
		segmentedText =  pst.tokenize(text)

		
		return segmentedText