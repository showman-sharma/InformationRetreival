from util import *

# Add your import statements here
import numpy as np
import csv


class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here
		cnt = 0
		for ID in query_doc_IDs_ordered[:k]:
			if ID in true_doc_IDs: cnt += 1
		precision = cnt/k
		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = -1

		#Fill in code here
		true_doc_IDs = []
		bad_query_results = []
		PrecisionSum = 0
		for ID in query_ids:
			true_doc_IDs = [int(qrel["id"]) for qrel in qrels if qrel["query_num"]==str(ID)]
			precision_current = self.queryPrecision(doc_IDs_ordered[ID-1], ID, true_doc_IDs, k)
			PrecisionSum += precision_current


            
		meanPrecision = PrecisionSum/(len(query_ids))
		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1

		#Fill in code here
		cnt = 0
		for i in true_doc_IDs:
			if (i in query_doc_IDs_ordered[:k]): cnt += 1
		recall = cnt/(len(true_doc_IDs))
		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1

		#Fill in code here
		true_doc_IDs = []
		RecallSum = 0
		for ID in query_ids:
			true_doc_IDs = [int(qrel["id"]) for qrel in qrels if qrel["query_num"]==str(ID)]
			RecallSum += self.queryRecall(doc_IDs_ordered[ID-1], ID, true_doc_IDs, k)
		meanRecall = RecallSum/(len(query_ids))

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		#Fill in code here
		Precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		Recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)

		if(Precision == 0 and Recall == 0):
			fscore = 0
		else:
			fscore = (2*Precision*Recall)/(Precision + Recall) # 2PR/(P+R)
		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1

		#Fill in code here
		true_doc_IDs = []
		FscoreSum = 0
		for i in query_ids:
			true_doc_IDs = [int(qrel["id"]) for qrel in qrels if qrel["query_num"]==str(i)]
			FscoreSum += self.queryFscore(doc_IDs_ordered[i-1], i, true_doc_IDs, k)
		meanFscore = FscoreSum/(len(query_ids))

		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = -1

		#Fill in code here
		true_doc_ids = true_doc_IDs[0]
		true_doc_relavances = true_doc_IDs[1]

		rel_list = []
        
        # DCG
		DCG = 0
		for i, doc_ID in enumerate(query_doc_IDs_ordered[:k]):
			if doc_ID in true_doc_ids:
				doc_idx = true_doc_ids.index(doc_ID)
				rel = true_doc_relavances[doc_idx]
				rel_list.append([rel,i+1])
				DCG += rel/(np.log2(i + 2))

		ideal_rel_list = list(sorted(rel_list, key = lambda item : item[0], reverse = True))
        
        # IDCG
		IDCG = 0
		for i in range(len(ideal_rel_list)):
			rel = ideal_rel_list[i][0]
			IDCG += rel/(np.log2(i + 2))

		# nDCG
		if IDCG == 0: nDCG = 0
		else:
			nDCG = DCG/IDCG
            
            
		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = -1

		#Fill in code here
		true_doc_IDs = []
		NDCGSum = 0
		nDCGdata = {}
		for i in query_ids:
			true_doc_IDs_ids = [int(qrel["id"]) for qrel in qrels if qrel["query_num"]==str(i)]
			true_doc_IDs_rels = [1/qrel["position"] for qrel in qrels if qrel["query_num"]==str(i)]     # relavances
			true_doc_IDs = [true_doc_IDs_ids, true_doc_IDs_rels]
			nDCG = self.queryNDCG(doc_IDs_ordered[i-1], i, true_doc_IDs, k)
			NDCGSum += nDCG
			nDCGdata[i] = nDCG
		meanNDCG = NDCGSum/(len(query_ids))

		# # field names 
		# fields = ['DOC ID','nDCG']
		# filename = 'nDCG'+str(k)+'.csv'
		# # writing to csv file 
		# with open(filename, 'w',newline='') as csvfile: 
		# 	# creating a csv writer object 
		# 	csvwriter = csv.writer(csvfile) 

		# 	# writing the fields 
		# 	csvwriter.writerow(fields) 

		# 	# writing the data rows 
		# 	csvwriter.writerows(nDCGdata)

		return meanNDCG, nDCGdata


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""
		avgPrecision = -1
		recall_old = 0
		#Fill in code here
		retrieved_k = query_doc_IDs_ordered[0:k]
		precision_sum = 0
		count = 0
		for l in range(k):
			if retrieved_k[l] in true_doc_IDs:
				recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, l+1)
				recall_diff = (recall-recall_old)
				precision_sum = precision_sum + self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, l+1)*recall_diff
				recall_old = recall
				count = count + recall_diff
		if count == 0:
			avgPrecision = 0
		else:
			avgPrecision = precision_sum*1.0/count

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = -1

		#Fill in code here
		true_doc_IDs = []
		AveragePrecisionSum = 0
		APdata = {}
		for i in query_ids:
			true_doc_IDs = [int(qrel["id"]) for qrel in q_rels if qrel["query_num"]==str(i)]
			AP = self.queryAveragePrecision(doc_IDs_ordered[i-1], i, true_doc_IDs, k)
			AveragePrecisionSum += AP
			APdata[i] = AP
		meanAveragePrecision = AveragePrecisionSum/(len(query_ids))

		return meanAveragePrecision, APdata

