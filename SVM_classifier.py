#Ubah data kedalam data frame
import pandas as pd
df=pd.read_csv("D:/Machine Learning/Coding/SVM/diabetes.csv")
print(df.head())
#cek data
print(df.info())
"""
 #   Column                    Non-Null Count  Dtype
---  ------                    --------------  -----
 0   Pregnancies               768 non-null    int64
 1   Glucose                   768 non-null    int64
 2   BloodPressure             768 non-null    int64
 3   SkinThickness             768 non-null    int64
 4   Insulin                   768 non-null    int64
 5   BMI                       768 non-null    float64        
 6   DiabetesPedigreeFunction  768 non-null    float64        
 7   Age                       768 non-null    int64
 8   Outcome                   768 non-null    int64

 data siap dipakai ->semuanya numerik
"""
#pisahkan dataset

#memisahkan atribut pada data set dan menyimpannya pada variabel
X=df[df.columns[:8]]#data pada columus 1-8

#memisahkan atribut pada data set dan menyimpannya pada variabel
y=df["Outcome"]

#skala pada data set tersebut berbeda
"""
   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome
0            6      148             72             35        0  33.6                     0.627   50        1
1            1       85             66             29        0  26.6                     0.351   31        0
2            8      183             64              0        0  23.3                     0.672   32        1
3            1       89             66             23       94  28.1                     0.167   21        0
4            0      137             40             35      168  43.1                     2.288   33        1
->standarisasi
"""
from sklearn.preprocessing import StandardScaler
#standarisasi 
scaler=StandardScaler()
scaler=scaler.fit_transform(X)
print(scaler)
"""
[[ 0.63994726  0.84832379  0.14964075 ...  0.20401277  0.46849198
   1.4259954 ]

 [-0.84488505 -1.12339636 -0.16054575 ... -0.68442195 -0.36506078
  -0.19067191]

 [ 1.23388019  1.94372388 -0.26394125 ... -1.10325546  0.60439732
  -0.10558415]
 ...

 [ 0.3429808   0.00330087  0.14964075 ... -0.73518964 -0.68519336
  -0.27575966]

 [-0.84488505  0.1597866  -0.47073225 ... -0.24020459 -0.37110101
   1.17073215]

 [-0.84488505 -0.8730192   0.04624525 ... -0.20212881 -0.47378505
  -0.87137393]]
"""
#pisahkan data set untuk train dan testing->hot decoder
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.30,random_state=42)

#membuat train dan test ->support vector classifier=>supervised
from sklearn.svm import SVC
clf=SVC()
#train
clf=clf.fit(X_train,Y_train)
#testing
nilai=clf.score(X_test,Y_test)
print("Hasil Prediksi model svm terhadap svc adalah ",nilai)
# print(clf)
#visualisasi
import matplotlib.pyplot as plt
import seaborn as sns
fig,ax=plt.subplots(figsize=(10,8))
sns.lineplot(X=list(range(1,11)), y=X_train)
plt.show()
plt.show()