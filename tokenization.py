from util import *

# Add your import statements here
from nltk.tokenize import TreebankWordTokenizer

from autocorrect import Speller

def decapitalize(s, upper_rest = False):
  return ''.join([s[:1].lower(), (s[1:].upper() if upper_rest else s[1:])])

class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""
        
		tokenizedText = []
		for sentence in text:
			words = sentence.strip().split()
			data = []
			for word in words:
				word = Word(word.replace("/", ""))
				data.append(word.correct())

			tokenizedText.append(data)
        
		#Fill in code herefrom autocorrect import Speller

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []
		for sentence in text:
			tokenizer = TreebankWordTokenizer()
			words = tokenizer.tokenize(sentence)
			decap = []
			for word in words:
				decap.append(decapitalize(word))
			tokenizedText.append(decap)
		#Fill in code here

		return tokenizedText