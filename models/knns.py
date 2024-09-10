class KNN:
  def __init__(self, k=10, distance_metric='euclidean'):
    self.k = k 
    self.distance_metric = distance_metric

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

  def predict(self, X):
    predicted_labes = [self]

  def forward(self, X_trn, y_trn, x):
      
    
