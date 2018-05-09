Stappen

# Stap 1: Installaties
pip install numpy<br>
pip install matplotlib<br>
pip install sklearn<br>

# Stap 2: depencencies
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm
```

# Stap 2: data
```python
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

# Labels die bij de punten horen, op volgorde, waarbij er maar twee mogelijke gevallen zijn
y = [0,1,0,1,0,1,1,0,0,0,1]

# Labels waarbij meerdere gevallen mogelijk zijn
# y = [0,1,0,1,0,2,1,3,3,3,2]
```

# Stap 3: Methode selecteren en coördinaten aan labels koppelen
```python
# Aangeven dat het lineair is(de classifier)
clf = svm.SVC(kernel='linear', C = 1.0)

# Features aan de labels koppelen
clf.fit(X,y)
```

# Stap 4: User input
```python
# X-coördinaat aan de gebruiker vragen
xC = input("Welke x-coördinaat wil je voorspellen?")
print("X-coördinaat: ", xC)

# Y-coördinaat aan de gebruiker vragen
yC = input("Welke y-coördinaat wil je voorspellen?")
print("X-coördinaat: ", yC)
```

# Stap 5: Voorspelling maken en deze printen
```python
# De voorspelling maken van de opgegeven coördinaten en deze printen
voorspelling = clf.predict([float(xC),float(yC)])
print("Voorspelling voor coördinaten: (",([float(xC),float(yC)]),"), de voorspelling valt binnen categorie ", voorspelling)
```

# Stap 6: Plot de grafiek
```python

```
