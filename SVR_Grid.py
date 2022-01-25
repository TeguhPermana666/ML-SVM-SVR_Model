import pandas as pd

#convert data to the data frame
data=pd.read_csv("D:/Machine Learning/Coding/SVM/Salary_Data.csv")
print(data.head())
#pisahkan atribut
X=data["YearsExperience"]
y=data["Salary"]

#mengubah bentuk ->jika hanya ada satu atribut pada data set
import numpy as np
X=X[:, np.newaxis]
print(X)
#membangun model dengan paramter C, gamma dan kernel 
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

model=SVR()
parameter={
    "kernel":["rbf"],
    "C":[1000.10000,100000],
    "gamma":[0.5,0.05,0.005]
}
grid_search=GridSearchCV(model,parameter)
#melatih model dengan funsi fit
grid_search=grid_search.fit(X,y)
# print(grid_search)
best=grid_search.best_params_
print(best)
modul_baru=SVR(kernel="rbf",C=100000,gamma=0.005)
#gamma=>tingkat kesalahan data
#C =>menyesuaikan penyebaran data pada garis linear dgn margin of erornya (gamma)
modul_baru=modul_baru.fit(X,y)
print(modul_baru)
#visualisasi
import matplotlib.pyplot as plt
plt.scatter(X,y)
plt.plot(X,modul_baru.predict(X))
plt.show()