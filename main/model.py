import numpy as np 


class TF_IDF:

	def __init__(self,texts):
		self.texts=texts
		self.vocab=self.get_vocab()
		
	def get_vocab(self):

		vocab=[]
		for doc in self.texts:
			for term in doc.split(" "):
				if term not in vocab:
					vocab.append(term)

		return vocab 

	def If(self,term,doc):

		term_list=doc.split(' ')
		existing_num=term_list.count(term)
		return existing_num/len(term_list)

	def idf(self,term):

		count=0
		for doc in self.texts:
			if term in doc:
				count+=1

		return np.log10(len(self.texts)/count)

	def TFIDF(self,term,doc):

		return self.If(term,doc)*self.idf(term)

	def transform(self,doc):
		result=[]
		for term in self.vocab:
			result.append(self.TFIDF(term,doc))
		return result
