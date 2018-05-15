import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm

# Coördinaten van de punten
X = np.array([[1,2],
             [5,8],
             [1.5,1.8],
             [6,9],
             [1,0.6],
             [9,11],
             [4,7],
             [12,5],
             [10,6],
             [13,4],
             [10,10]])

# Labels die bij de punten horen, op volgorde
# y = [0,1,0,1,0,2,1,3,3,3,2]

# De coördinaten alleen maar gelabeld op 0 en 1
y = [0,1,0,1,0,1,1,0,0,0,1]

# Aangeven dat het lineair is(de classifier)
clf = svm.SVC(kernel='linear', C = 1.0)

# Features aan de labels koppelen
clf.fit(X,y)

# X-coördinaat aan de gebruiker vragen
xC = input("Welke x-coördinaat wil je voorspellen?")
print("X-coördinaat: ", xC)

# Y-coördinaat aan de gebruiker vragen
yC = input("Welke y-coördinaat wil je voorspellen?")
print("Y-coördinaat: ", yC)

# De voorspelling maken van de opgegeven coördinaten en deze printen
voorspelling = clf.predict([[float(xC),float(yC)]])
print("Voorspelling voor coördinaten: (",([float(xC),float(yC)]),"), de voorspelling valt binnen categorie ", voorspelling)

w = clf.coef_[0]

a = -w[0] / w[1]

xx = np.linspace(0,15)
yy = a * xx - clf.intercept_[0] / w[1]

# De schijdingslijn, kan uitgezet worden door deze in comments te plaatsen
h0 = plt.plot(xx, yy, 'k-', label="Schijdingslijn")

# Het tekeken van de grafiek, kan ook uitgezet worden
plt.scatter(X[:, 0], X[:, 1], c = y)
plt.show()
