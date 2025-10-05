from __future__ import division, unicode_literals
from collections import Counter
import numpy as np

class DocVector:
	
	bloblist = []
	vectorKeywordIndex = {}
	weightOpt = ''

	def __init__(self, bloblist = [], vectorKeywordIndex = {}, weightOpt = 'TFIDF'):
		self.bloblist = bloblist
		self.vectorKeywordIndex = vectorKeywordIndex
		self.weightOpt = weightOpt

	def buildTfVector(self, blob, vectorKeywordIndex, weightOpt):
		# Initialise vector with 0's
		tfVector = np.zeros(len(vectorKeywordIndex))
		wordCount = Counter(blob.words)  # Count occurrences of each word
		for word, count in wordCount.items():  # For each word in the document
			if word in vectorKeywordIndex:
				tfVector[vectorKeywordIndex[word]] = count	# raw TF
				# print(word, " : ", tfVector[vectorKeywordIndex[word]])
		maxCount = np.max(tfVector)
		if weightOpt == 'TFIDF' and maxCount > 0:
			return tfVector / maxCount # return normalised tfVector
		return tfVector # return raw TF

	def buildIdfVector(self, bloblist, vectorKeywordIndex):
		docCount = len(bloblist) # #documents
		wordDocFreq = Counter(word for blob in bloblist for word in set(blob.words)) # #documents containing word
		idfVector = np.zeros(len(vectorKeywordIndex))
		for word, count in wordDocFreq.items():
			if word in vectorKeywordIndex:
				idfVector[vectorKeywordIndex[word]] = np.log(docCount / (1 + count))
		return idfVector

	def buildIfidfVector(self, tfVector, idfVector, weightOpt):
		if weightOpt == 'RawTF':
			return tfVector
		return tfVector * idfVector