from util import *

# Add your import statements here
from collections import Counter
from math import log
import numpy as np
#from scipy import spatial
from math import sqrt
import time
import matplotlib.pyplot as plt
import csv

def decapitalize(s, upper_rest = False):
  return ''.join([s[:1].lower(), (s[1:].upper() if upper_rest else s[1:])])

class InformationRetrieval():

	def __init__(self):
		self.index = {}
		self.N = 0
		self.idf = {}
		self.docVecs = {}
		self.Uinv = None

	def buildIndex(self, docs, docIDs,docTitles):
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

		#BUilding theindex
		print('Begin building Index:')
		N = len(docIDs)
		self.N = N
		
		t1 = time.perf_counter()
		for i in range(N):
			for sentence in docs[i]:
				l = len(sentence)	
			
				for j in range(l-1):
					biword = sentence[j]+'-'+sentence[j+1]
					if biword in index:
						index[biword].append(docIDs[i])
					else:
						index[biword] = [docIDs[i]]
					word = sentence[j]
					if word in index:
						index[word].append(docIDs[i])
					else:
						index[word] = [docIDs[i]]
				if l>0:
					word = sentence[l-1]
					if word in index:
						index[word].append(docIDs[i])
					else:
						index[word] = [docIDs[i]]	

			docTitle = docTitles[i]
			l = len(docTitle)
			for j in range(l-1):
				biword = docTitle[j]+'-'+docTitle[j+1]
				if biword in index:
					index[biword].append(docIDs[i])
				else:
					index[biword] = [docIDs[i]]
				word = docTitle[j]
				if word in index:
					index[word].append(docIDs[i])
				else:
					index[word] = [docIDs[i]]		
			if l>0:
				word = docTitle[l-1]
				if word in index:
					index[word].append(docIDs[i])
				else:
					index[word] = [docIDs[i]]			
				
		for t in index:
			c = Counter(index[t])
			if len(c)<0.5*N and len(index[t])>2:
				self.index[t] = c	

		print('# words+bigrams used = ',len(self.index))
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

		
		# # #LSA
		# t1 = time.perf_counter()
		# A = np.array([self.docVecs[docID] for docID in self.docVecs]).T
		# u,s,v = np.linalg.svd(A)
		# plt.plot(s)
		# plt.show()
		# ndim = 500
		# # self.Uinv = u[:,:500].T#np.linalg.pinv(u[:,:500])
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
			unigrams = [(decapitalize(str(q))) for q in query[0]]
			query = [unigrams[i]+'-'+unigrams[i+1] for i in range(len(unigrams)-1)] + unigrams
			c = Counter(query)	
			Qvec = np.array([c[t]*self.idf[t]/len(query) if t in c else 0 for t in self.index])
			# Qvec = self.Uinv@Qvec			

			# sim = {docID:(np.dot(Qvec,self.docVecs[docID])/uuvv[docID] if uuvv[docID]>0 else 0)   for docID in self.docVecs}	
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
		filename = "execTime_bigram2.csv"

		# writing to csv file 
		with open(filename, 'w',newline='') as csvfile: 
			# creating a csv writer object 
			csvwriter = csv.writer(csvfile) 
			
			# writing the fields 
			csvwriter.writerow(fields) 
			
			# writing the data rows 
			csvwriter.writerows(query_exec_times)

	
		return doc_IDs_ordered




