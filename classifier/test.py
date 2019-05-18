import numpy as np
from sklearn import linear_model

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
Y = np.array([4, 3, 2, 1])
clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
clf.fit(X, Y)

print(clf.predict([[-0.8, -1]]))