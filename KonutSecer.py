# -*- coding: utf-8 -*-
"""
Accuracy Rate = %60
"""

import numpy as np
from sklearn.model_selection import train_test_split

X = np.array([
    [5,5,1,4,1,5],
    [5,5,1,5,1,5],
    [4,2,2,4,1,1],
    [1,2,1,2,5,4],
    [3,5,5,5,3,5],
    [3,4,4,4,1,5],
    [4,2,4,2,1,5],
    [3,4,2,5,4,2],
    [3,5,2,4,1,5],
    [4,5,1,5,2,5],
    [4,4,2,5,1,4],
    [1,5,2,5,4,5],
    [3,5,2,5,3,4],
    [4,4,4,4,1,5],
    [3,4,4,2,1,5],
    [1,5,4,5,5,5],
    [3,2,2,2,5,4],
    [3,5,2,4,1,4],
    [1,5,4,5,4,5],
    [3,5,1,5,5,4],
    [3,4,4,5,2,5],
    [3,4,5,5,3,5],
    [3,5,5,5,5,5],
    [4,4,2,4,1,5],
    [3,4,4,5,1,5],
    [3,4,4,5,1,5],
    [5,5,1,5,1,5],
    [5,5,2,5,3,5],
    [5,2,1,2,4,5],
    [5,5,5,5,3,5]

])
Y = np.array([
       0,
    0,
    0,
    0,
    1,
    1,
    0,
    0,
    0,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    1

])



X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify = Y
)
print("X Train: ", X_train.shape)
print("X Test: ", X_test.shape)
print("Y Train: ", Y_train.shape)
print("Y Test: ", Y_test.shape)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

k_degerleri = range(1,11)
accuracies = []

for k in k_degerleri:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    accuracies.append(accuracy)
    print(f"K = {k}, Doğruluk: {accuracy}")

en_iyi_k = k_degerleri[accuracies.index(max(accuracies))]
print(f"En iyi K değeri: {en_iyi_k}")
#

import matplotlib.pyplot as plt

plt.figure()
plt.plot(k_degerleri, accuracies, marker="o")
plt.xlabel("K Değeri")
plt.ylabel("Doğruluk")
plt.title("KNN Doğruluk Değerleri")
plt.xticks(k_degerleri)
plt.grid(True)
plt.show()

knn_final = KNeighborsClassifier(n_neighbors=en_iyi_k)
knn_final.fit(X_train, Y_train)

print("Yeni ev için 6 özelliği gir:")

f1 = int(input("KonumPuanı (1-5 arası): "))
f2 = int(input("Net Alan: "))
f3 = int(input("Kat: "))
f4 = int(input("Oda Sayısı(Salon+Odalar Toplamı): "))
f5 = int(input("Fiyat (TL): "))
f6 = int(input("Bina Yaşı: "))
if(90 < f2 < 120):
    f2 = 5
elif(70 < f2 < 89 or 121 < f2 < 150):
    f2 = 4
elif(50 < f2 < 69 or f2 > 150):
    f2 = 2
else:
    f2 = 1

if(f3 == 1 or f3 == 2):
    f3 = 5
elif(f3 == 3 or f3 == 4):
    f3 = 4
elif(f3==0):
    f3 = 3
elif(5 <= f3 <= 6):
    f3 = 2
else:
    f3 = 1

if(f4 == 3):
    f4 = 5
elif(f4 == 2):
    f4 = 2
elif(f4 == 4):
    f4 = 4
elif(f4 == 1):
    f4 = 1
elif(f4 >= 5):
    f4 = 2
else:
    f4 = 1

if(f5 <= 2000000):
    f5 = 5
elif(f5 < 3000000):
    f5 = 4
elif(f5 < 4000000):
    f5 = 3
elif(f5 < 5000000):
    f5 = 2
else:
    f5 = 1

if(f5 <= 5):
    f5 = 5
elif(f5 <= 10):
    f5 = 4
elif(f5 <= 15):
    f5 = 3
elif(f5 <= 20):
    f5 = 2
else:
    f5 = 1


yeni_ev = np.array([[f1, f2, f3, f4, f5, f6]])

tahmin = knn_final.predict(yeni_ev)
print("Tahmin: ",tahmin)
if tahmin[0] == 1:
    print("Satın alabilirsiniz")
else:
    print("Satın alamazsınız")
