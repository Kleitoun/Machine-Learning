import torch

class Perceptron:
	def __init__(self, K=1000, D=None, bias=1):
		if D is None:
			raise ValueError('D must be provided')

		self.max_updates = K
		X = torch.tensor(D[:,:-1], dtype=torch.float32)
		self.y = torch.tensor(D[:,-1], dtype=torch.float32)
		self.n = X.shape[0]
		self.d = X.shape[1]


		self.weights = torch.rand(d+1, dtype=torch.float32)

		ones = torch.ones(self.n,1)
		self.X = torch.cat((X,ones),dim=1)

	def fit(self):
		k = 0
		while k < self.max_updates:
			misclassified=False
			for i in range(self.n):
				if (self.X[i]@self.weights)*self.y[i] <= 0:
					self.weights += (self.y[i]*self.X[i])
					misclassified = True
					k+=1
			if not misclassified:
				break

		return self.weights

	def predict(self, X):
		ones = torch.ones(X.shape[0], 1)
		X = torch.cat((X,ones), dim=1)
		return torch.sign(X@self.weights)


			


