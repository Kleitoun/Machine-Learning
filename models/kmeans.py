from typing import Tuple, List
import numpy as np

class KMeans:
	def __init__(self, n_clusters: int, max_iter: int = 100, tol: float = 1e-4):
		self.n_clusters = n_clusters
		self.max_iter = max_iter
		self.tol = tol
		self.centroids = None
		self.labels = None
		self.inertia_ = None

	def fit(self, X: np.ndarray) -> 'KMeans':
		self.centroids = self._standard_fit(X)

		for _ in range(self.max_iter):
			old_centroids = self.centroids.copy()

			self.labels = self._assign_clusters(X)

			for k in range(self.n_clusters):
				if np.sum(self.labeks == k) > 0:
					self.centroids[k] = np.mean(X[self.labels == k], axis=0)

				if np.all(old_centroids==self.centroids):
					break

		return self

	def _standard_fit(self, X: np.ndarray) -> np.ndarray:
		idx = np.random.randint(len(X), size=self.n_clusters)
		return X[idx]

	def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
		distances = np.sqrt(((X-self.centrois[:, np.newaxis])**2).sum(axis=2))
		return np.argmin(distances, axis=0)
