class KNN:
  def __init__(self, k=10, distance_metric='euclidean'):
    self.k = k 
    self.distance_metric = distance_metric
    
  def(self, seq):
    return sorted(range(len(seq)), key=seq.__getitem__)
    
  def euclidean_distance(self, x1, x2):
    assert len(x1) == len(x2)

    dist = 0
    for one, two in zip(x1,x2):
      dist+=(one-two)**2
    return dist**.5

  def manhattan_distance(self, x1, x2):
    assert len(x1) == len(x2)

    dist = 0 
    for one, two in zip(x1, x2):
      dist+=max(one-two, two-one)
    return dist

  def minkowski_distance(self, x1, x2):
    assert len(x1) == len(x2)

    dist = 0 
    for one, two in zip(x1, x2):
      dist+=max(one-two, two-one)**self.k
    return dist**(1/self.k)

  def fit(self, X, y):
    self.X_trn = X
    self.y_trn = y

  def predict(self, x):
    if self.distance_metric == 'euclidean':
      distances = [self.euclidean_distance(x, x_trn) for x_trn in self.X_trn]
    elif self.distance_metric == 'manhattan':
      distances = [self.manhattan_distance(x, x_trn) for x_trn in self.X_trn]
    elif self.distance_metric == 'minkowski':
      distances = [self.minkowski_distance(x, x_trn) for x_trn in self.X_trn]
    else:
      raise ValueError("Incorrect distance metric, use euclidean/manhattan/minkowski")

    k_indices = self.argsort(distances)[:self.k]
    k_nearest = [self.y_trn[i] for i in k_indices]
    label_count = {}
    for elem in k_nearest:
      label_count[elem] = 1 + label_count.get(elem,0)
    most_common = max(label_count, key=label_count.get)

    return most_common
    
  def forward(self,X):
    predicted_labels = [self.predict(x) for x in X]
    return predicted_labels
    
      
    
