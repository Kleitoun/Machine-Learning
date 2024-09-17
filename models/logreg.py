import torch
import math

class LogisticRegression:
	def __init__(self, data=None, threshold=None, iterations=None, lr=None):
		if not data:
			raise ValueError('No data provided')

		self.n = len(data.shape[0])
		self.m = len(data.shape[1])
		X = torch.tensor(data[:,:-1])
		bias = torch.ones(self.n,1)
		self.X = torch.cat((X, bias), dim=1)
		self.y = torch.tensor(data[:,-1])
		self.weights = torch.rand(self.n+1)
		self.lr = lr if lr is not None else 0.01
		self.threshold = threshold if threshold is not None else 0.5
		self.iterations = iterations if iterations is not None else 1000

	def _sigmoid(self, z):
		z = torch.tensor(z)
		return 1/(1+torch.exp(-z))

	def _log_likelihood(self):
		z = torch.dot(self.X, self.weights.T)
		ll = torch.sum(self.y*z+np.log(1+np.exp(z)))
		return ll

	def fit(self, max_iter=self.iterations, lr=self.lr, tolerance=0.0005):
		iter = 0
		while iter < max_iter:
			z = torch.dot(self.features, self.weights)
			preds = self._sigmoid(z)
			error = self.y - preds
			gradient = torch.dot(self.features.T, error)
			self.weights += lr*gradient
			iter+=1




