from util import *

# Add your import statements here
# from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# from nltk.stem.snowball import SnowballStemmer
#import nltk
#nltk.download("wordnet")
#nltk.download('omw-1.4')




class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		#ps = PorterStemmer()
		lemmatizer = WordNetLemmatizer()
		# stemmer = SnowballStemmer(language='english')

		reducedText = []

		#Fill in code here
		for tokens_list in text:
			l = []
			for token in tokens_list:
				# l.append(stemmer.stem(token))
				new_toks = token.split('-')
				for tok in new_toks:
					l.append(lemmatizer.lemmatize(tok))
			reducedText.append(l)

		
		return reducedText


