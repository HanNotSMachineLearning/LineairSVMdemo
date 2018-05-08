import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm

# Coördinaten van de punten
X = np.array([[1,2],
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11],
             [2,6],
             [3,5],
             [10,6],
             [13,4],
             [10,10]])

# Labels die bij de punten horen, op volgorde
y = [0,1,0,1,0,1,1,0,0,0,1]

# Aangeven dat het lineair is
clf = svm.SVC(kernel='linear', C = 1.0)

clf.fit(X,y)
print(clf.predict([0.58,0.76]))

print(clf.predict([10.58,10.76]))

w = clf.coef_[0]
print(w)

a = -w[0] / w[1]

xx = np.linspace(0,15)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="Schijdingslijn")

plt.scatter(X[:, 0], X[:, 1], c = y)
plt.legend()
plt.show()
