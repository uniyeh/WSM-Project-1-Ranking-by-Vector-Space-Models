import sys

#http://www.scipy.org/
try:
	import numpy as np
	from numpy.linalg import norm
except:
	print("Error: Requires numpy from http://www.scipy.org/. Have you installed scipy?")
	sys.exit() 

def removeDuplicates(list):
	""" remove duplicates from a list """
	return set((item for item in list))


def cosine(vector1, vector2):
	""" related documents j and q are in the concept space by comparing the vectors :
		cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
	norm_product = norm(vector1) * norm(vector2)
	if norm_product == 0:
		return 0.0
	return float(np.dot(vector1,vector2) / norm_product)

def Euclidean(vector1, vector2):
	""" calculate the euclidean distance between two vectors """
	return np.linalg.norm(np.array(vector1) - np.array(vector2))

def average(list):
	""" return the average value of a list """
	return np.mean(np.array(list))

def total(list):
	return np.sum(np.array(list))