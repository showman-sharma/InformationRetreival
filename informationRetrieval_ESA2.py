from util import *

# Add your import statements here
from collections import Counter
from math import log
import numpy as np
#from scipy import spatial
from math import sqrt
import time
import csv
import wikipedia
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
# from nltk.stem.snowball import SnowballStemmer
import math
import matplotlib.pyplot as plt
# import progressbar
from tqdm import tqdm

import json


def decapitalize(s, upper_rest = False):
  return ''.join([s[:1].lower(), (s[1:].upper() if upper_rest else s[1:])])

class InformationRetrieval():

	def __init__(self):
		self.index = {}
		self.N = 0
		self.idf = {}
		self.docVecs = {}
		self.Uinv = None
		self.ESAmatrix = None	

	def buildIndex(self, docs, docIDs, docTitles):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""
		index = {}
			

		#Fill in code here

		#Building the word index
		print('Begin building Index:')
		N = len(docIDs)
		self.N = N
		
		t1 = time.perf_counter()
		for i in range(N):
			for sentence in docs[i]:
				for word in sentence:
					if word in index:
						index[word].append(docIDs[i])
					else:
						index[word] = [docIDs[i]]
			for word in docTitles[i]:
				if word in index:
					index[word].append(docIDs[i])
				else:
					index[word] = [docIDs[i]]			

				
		for t in index:
			c = Counter(index[t])
			if len(c)<0.5*N and len(index[t])>1:
				self.index[t] = c	
		print('# Words used = ',len(self.index))
		t2 = time.perf_counter()
		print('Time taken to form indexes = {} minutes'.format((t2-t1)/60))


		#Building the idf
		t1 = time.perf_counter()
		self.idf = {t: log(N/len(self.index[t].keys())) for t in self.index}
		t2 = time.perf_counter()
		print('Time taken to form idf = {} minutes'.format((t2-t1)/60))

		#Building tf-Idf vectors for each doc
		t1 = time.perf_counter()
		self.docVecs = {docIDs[i]: np.array([0]*len(self.index.keys()) if len(docs[i])==0 else [self.index[t][docIDs[i]]*self.idf[t]/len(docs[i]) for t in self.index])	for i in range(N)}	
		t2 = time.perf_counter()
		print('Time taken to form tfidf vectors= {} minutes'.format((t2-t1)/60))

		#ESA vectors
		t1 = time.perf_counter()
		vocab = list(self.index.keys())
				
		self.ESAmatrix = self.ESA(docTitles,vocab)
		self.docVecs = {docID:np.dot(self.docVecs[docID],self.ESAmatrix) for docID in self.docVecs}
		t2 = time.perf_counter()
		print('Time taken to form ESA matrix= {} minutes'.format((t2-t1)/60))

		# # #LSA
		# t1 = time.perf_counter()
		# A = np.array([self.docVecs[docID] for docID in self.docVecs]).T
		# u,s,v = np.linalg.svd(A)
		# # plt.plot(s)
		# # plt.show()
		# ndim = 350
		# self.Uinv = np.diag(1/s[:ndim])@u[:,:ndim].T
		# for docID in self.docVecs:
		# 	self.docVecs[docID] =  self.Uinv@self.docVecs[docID]
		# t2 = time.perf_counter()
		# print('Time taken to form LSA vectors= {} minutes'.format((t2-t1)/60))
		

		#normalizing tfidf vectors
		t1 = time.perf_counter()
		uuvv = {docID:(np.linalg.norm(self.docVecs[docID])) for docID in self.docVecs}	
		for docID in self.docVecs:
			if uuvv[docID] > 1e-6:
				factor = 1/uuvv[docID]
			else:
				factor = 1e6
			self.docVecs[docID] = self.docVecs[docID]*factor
		t2 = time.perf_counter()
		print('Time taken to normalize tfidf vectors= {} minutes'.format((t2-t1)/60))

	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""
		#print(self.index)

		doc_IDs_ordered = []
		query_exec_times = []

		#Fill in code here\

		t1 = time.perf_counter()

		
		for query in queries:
			t_start = time.perf_counter()
			query = [decapitalize(str(q)) for q in query[0]]
			c = Counter(query)	
			Qvec = np.array([c[t]*self.idf[t]/len(query) if t in c else 0 for t in self.index])
			Qvec = (Qvec@self.ESAmatrix)
			# Qvec = self.Uinv@	Qvec
			sim = {docID:np.dot(Qvec,self.docVecs[docID]) for docID in self.docVecs}	
			doc_IDs_ordered.append(sorted(sim, key=sim.get, reverse = True))
			t_stop = time.perf_counter()
			exec_time = t_stop-t_start
			query_exec_times.append([exec_time])
		t2 = time.perf_counter()
		print('Time taken over queries= {} minutes'.format((t2-t1)/60))

		# field names 
		fields = ['Exec time'] 		

		# name of csv file 
		filename = "execTime_ESA2.csv"

		# writing to csv file 
		with open(filename, 'w',newline='') as csvfile: 
			# creating a csv writer object 
			csvwriter = csv.writer(csvfile) 
			
			# writing the fields 
			csvwriter.writerow(fields) 
			
			# writing the data rows 
			csvwriter.writerows(query_exec_times)


	
		return doc_IDs_ordered


	def ESA(self,docTitles,vocab):
		try:
			with open('ESAmatrix.npy','rb') as f:
				ESAmatrix = np.load(f)
				return ESAmatrix
		except:
			pass
		try:
			with open('wikiData.json') as json_file:
				summary = json.load(json_file)
			#Building the word index
			vocab_len = len(vocab)
			N = len(summary)
			procSummary = []

			t1 = time.perf_counter()
			index = {v:[] for v in vocab}
			wikidata = {}
			concepts = []
			print('\nParsing through summaries (preprocessed)')
			for i in tqdm(range(N)):
				concept = summary[str(i)]['concept']
				if concept not in concepts:
					sentence = (summary[str(i)]["summary"])
					for word in sentence:
						if word in index:
							index[word].append(i)
					procSummary.append(sentence)
					concepts.append(concept)
			N = len(procSummary)




		except:	
			#extract data from wikipedia
			t1 = time.perf_counter()
			concepts = []
			i = 0;
			print('\nParsing through document titles:')
			for title in tqdm(docTitles):
				try:
					concepts = concepts + wikipedia.search(title)

				except:
					continue

			summary = []
			links = []
			concepts_new = []


			print('\nParsing through concepts:')

			for concept in tqdm(concepts):
				try:
					p = wikipedia.page(concept)
					concepts_new.append(concept)
					summary.append(p.summary)

				except:
					pass

				

			t2 = time.perf_counter()
			print('Time taken to extract Wikipedia data = {} minutes'.format((t2-t1)/60))
			
			
			#Building the word index
			vocab_len = len(vocab)
			N = len(summary)
			procSummary = []
			# stemmer = SnowballStemmer(language='english')
			lemmatizer = WordNetLemmatizer()

			t1 = time.perf_counter()
			index = {v:[] for v in vocab}
			wikidata = {}
			print('\nParsing through summaries')
			for i in tqdm(range(N)):
				Sum = []
				for word in TreebankWordTokenizer().tokenize(summary[i]):
					# word = stemmer.stem(word)
					word = lemmatizer.lemmatize(word)
					Sum.append(word)
					if word in index:
						index[word].append(i)
				wikidata[i] = {'concept':concepts_new[i],'summary':Sum}
				procSummary.append(Sum)

			out_file = open("wikiData.json", "w")
			json.dump(wikidata, out_file, indent = 6) 
			out_file.close() 

		for t in index:
			c = Counter(index[t])
			index[t] = c	
		t2 = time.perf_counter()
		print('Time taken to form ESA index = {} minutes'.format((t2-t1)/60))


		#Building the idf
		t1 = time.perf_counter()

		idf = {t: log(N/(len(index[t].keys())+1))+1for t in index}
		t2 = time.perf_counter()
		print('Time taken to form idf in ESA = {} minutes'.format((t2-t1)/60))

		#Building tf-Idf matrix
		t1 = time.perf_counter()
		tf_idf_matrix = np.array([np.array([0]*vocab_len if len(procSummary[i])==0 else [index[t][i]*idf[t]/len(procSummary[i]) for t in index])	for i in range(N)])	
		t2 = time.perf_counter()
		print('Time taken to form ESA tfidf vectors= {} minutes'.format((t2-t1)/60))


		with open('ESAmatrix.npy','wb') as f:
			np.save(f, tf_idf_matrix.T)

		return tf_idf_matrix.T





				
		


