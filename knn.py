import numpy as np


"""
KNN steps:
1. calculate distance
2. get nearest neighbors
3. make predictions
"""

"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch
"""

import numpy as np
import pandas as pd
import math


class KNNLearner(object):
	def __init__(self, k):
		self.k = k # move along, these aren't the drones you're looking for

	def euclideanDistance(self, instance1, instance2):
		distance=0
		for x in range(len(instance1)):
			distance+= pow((instance1[x]-instance2[x]),2)
		return math.sqrt(distance)

	def euclideanDistanceB(self, point, dataSet):
		distance_np=np.empty(len(dataSet))
		for i in range(len(dataSet)):
			instance=dataSet[i]
			dist = np.linalg.norm(point-instance)
			distance_np[i]=dist
		return distance_np

	def getNeighbors(self, trainingSetX, trainingSetY, testInstance):
		trainingSize=trainingSetX.shape[0]
		distances=np.zeros(trainingSize)
		df_distances=pd.DataFrame(distances, index=np.arange(len(distances)),columns=['distance'])
		for i in range(0,trainingSize):
			df_distances.iloc[i]=self.euclideanDistance(trainingSetX[i],testInstance)
		df_sorted_distance=df_distances.sort(['distance'],ascending=[1])
		df_sorted_distance=df_sorted_distance[df_sorted_distance['distance']>0]
		first_k_index=df_sorted_distance.index[0:self.k]
		sum=0
		for each in first_k_index:
			sum+=trainingSetY[each]
		return sum/self.k

	def getNeighborsB(self, trainingSetX, trainingSetY, testSet):
		predY=[]
		for each in testSet:
			testpoint=each
			distance_np=self.euclideanDistanceB(each, trainingSetX)
			first_k_index=np.argsort(distance_np)
			for i in first_k_index:
				sum=0
				sum+=trainingSetY[i]
			predY.append(sum/self.k)
		return predY

	def addEvidence(self, dataX, dataY):
		"""
		@summary: Add training data to learner
		@param dataX: X values of data to add
		@param dataY: the Y training values
		"""
		'''
		# slap on 1s column so linear regression finds a constant term
		newdataX = np.ones([dataX.shape[0],dataX.shape[1]+1])
		newdataX[:,0:dataX.shape[1]]=dataX
		'''
		self.dataX, self.dataY=dataX, dataY

	def query(self,points):
		"""
		@summary: Estimate a set of test points given the model we built.
		@param points: should be a numpy array with each row corresponding to a specific query.
		@returns the estimated values according to the saved model.
		"""
		predY=[]
		for each in points:
			testpoint=each
			distance_np=self.euclideanDistanceB(each, self.dataX)
			first_k_index=np.argsort(distance_np)[0:self.k]
			sum=0
			for i in first_k_index:
				sum+=self.dataY[i]
			predY.append(sum/self.k)
		return predY


if __name__=="__main__":
	print "the secret clue is 'zzyzx'"
