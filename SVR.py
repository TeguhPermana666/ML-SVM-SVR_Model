"""
1.Ubah data menjadi Dataframe.
2.Pisahkan atribut dan label.
3.Latih model SVR.
4.Buat plot dari model.
"""
import pandas as pd

#membaca dataset dan mengubahnya menjadi dataframe
data=pd.read_csv("D:/Machine Learning/Coding/SVM/Salary_Data.csv")
#melihat data apakah semuanya numerik
print(data.info())
"""
 #   Column           Non-Null Count  Dtype
---  ------           --------------  -----
 0   YearsExperience  30 non-null     float64
 1   Salary           30 non-null     float64
"""
print(data.head())
#pisahkan atribut dengan label
import numpy as np
#pisahkan atribut dan label
X=data["YearsExperience"]
y=data["Salary"]
#mengubah bentuk atribut
X=X[:,np.newaxis]
print(X)
from sklearn.svm import SVR
#membangun model dengan parameter C,gamma,kernel
model=SVR(C=1000,gamma=0.05,kernel="rbf")
#train model
model=model.fit(X,y)
#Testing
hasil=model.score(X,y)
print(hasil)
#visualisasi
import matplotlib.pyplot as plt
model=model.predict(X)
fig,ax=plt.subplots()
ax.scatter(X,y)
ax.plot(X,model)
plt.show()