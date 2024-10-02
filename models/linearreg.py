import torch
import torch.nn as nn

class LinearRegression:
	def __init__(self, eta=0.001, n_iter=20):
		self.eta = eta
		self.iter = n_iter

	def fit(self, X, y):
		self.w = np.zeros(1+X.shape[1])
		self.cost = []
		self.m, self.n = X.shape

		for i in range(self.iter):
			output = X.dot(elf.w[1:]) + self.w[0]
			dW = - (2*self.X.T.dot(self.y - self.output))/self.m
			db = -2*np.sum(self.y-output)/self.m
			self.w[1:] -= self.eta * dW
			self.w[0] -= self.eta * db
			cost = (errors**2).sum() / 2.0
			self.cost.append(cost)

		return self

	def predict(self, X):
		return X.dot(self.w[1:]) + self.w[0]
