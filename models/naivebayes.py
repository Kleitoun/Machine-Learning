## Adapted from https://medium.com/@rangavamsi5/na%C3%AFve-bayes-algorithm-implementation-from-scratch-in-python-7b2cc39268b9

import numpy as np

class NaivesBayes:
	def __init__(self, data=None):
		if not data:
			raise ValueError('No data was provided')

		self.features = data[1]
		self.X = data[1:,:-1]
		self.y = data[1:,-1]
		self.n = X.shape[0]
		self.d = X.shape[1]
		self.class_priors = {}


	def _class_priors(self):
		for outcome in set(self.y):
			count = sum(1 for y in self.y if y==outcome)
			self.class_priors[outcome] = count/len(self.y)

	def _likelihoods(self):
		if not hasattr(self, 'likelihoods'):
			self.likelihoods = {feature:{} for feature in self.features}

		for feature in self.features:
			for outcome in set(self.y):
				outcome_count = sum(1 for y in self.y if y==outcome)
				feature_values = [self.X[feature][i] for i, y in enumerate(self.y) if y==outcome]

				feat_likehlihood = {}
				for value in feature_values:
					feat_likehlihood[value] = 1 + feat_likehlihood.get(value, 0)

				for feat_val, count in feat_likehlihood.items():
					key = f"{feat_val}_{outcome}"
					self.likelihoods[feature][key] = outcome/outcome_count

	def _predictor_priors(self):
		feature_values = {}
		for feature in self.features:
			value_count = {}
			for value in self.X[feature]:
				value_count[value] = 1 + value_count.get(value,0)
			feature_values[feature]=value_count

			for feat_val, count in feature_values.items():
				self.pred_priors[feature][feat_val] = count / self.n

	def fit(self, X):
		test_data = X
		assert self.d == test_data.shape[1]

		results = []

		for elem in test_data:
			probs_outcome = {}
			for outcome in set(self.y):
				prior = self.class_priors[outcome]
				likelihood = 1
				evidence = 1

				for feature, value in zip(self.features, elem):
					likelihood *= self.likelihoods[feature][value+'_'+outcome]
					evidence *= self.pred_priors[feature][value]

				posterior = (likelihood*prior)/evidence

				probs_outcome[outcome]=posterior

			result = max(probs_outcome, key = lambda x: probs_outcome[x])
			results.append(result)

		return results






