import torch

class Perceptron:
	def __init__(self, K=1000, D=None, bias=1):
		self.max_updates = K
		self.X = torch.tensor(D[:,:-1])
		self.y = torch.tensor(D[:,-1])
		self.n = len(D[0])
		self.weights = torch.rand(n)
		# self.bias = torch.tensor(bias)
		# self.params = torch.cat((self.weights, self.bias))

	def fit(self):
		k = 0
		misclassified=True
		while misclassified and k < self.K:
			misclassified=False
			for i in range(self.n):
				if (self.weights.T@self.X[i])*self.y[i] <= 0:
					self.weights = self.weights + (self.y[i]*self.X[i])
					k+=1

		return self.weights


			


