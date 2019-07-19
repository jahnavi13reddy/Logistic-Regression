

Import libraries
#scipy used for scientific calculation
import pandas as pd
import numpy as np #deals numbers and calculation scipy is also the same
import matplotlib.pyplot as plt
import seaborn as sns
#bring backend activities to front end
%matplotlib inline
​
Set warnings
import warnings
warnings.filterwarnings('ignore')
​
Set local path to access data
cd C:\Users\Jahnavi\Downloads\machine learning
C:\Users\Jahnavi\Downloads\machine learning
Load the data set
Iris=pd.read_csv('Iris.csv')
Intrepreting the data
Iris.head(20)
Id	SepalLengthCm	SepalWidthCm	PetalLengthCm	PetalWidthCm	Species
0	1	5.1	3.5	1.4	0.2	Iris-setosa
1	2	4.9	3.0	1.4	0.2	Iris-setosa
2	3	4.7	3.2	1.3	0.2	Iris-setosa
3	4	4.6	3.1	1.5	0.2	Iris-setosa
4	5	5.0	3.6	1.4	0.2	Iris-setosa
5	6	5.4	3.9	1.7	0.4	Iris-setosa
6	7	4.6	3.4	1.4	0.3	Iris-setosa
7	8	5.0	3.4	1.5	0.2	Iris-setosa
8	9	4.4	2.9	1.4	0.2	Iris-setosa
9	10	4.9	3.1	1.5	0.1	Iris-setosa
10	11	5.4	3.7	1.5	0.2	Iris-setosa
11	12	4.8	3.4	1.6	0.2	Iris-setosa
12	13	4.8	3.0	1.4	0.1	Iris-setosa
13	14	4.3	3.0	1.1	0.1	Iris-setosa
14	15	5.8	4.0	1.2	0.2	Iris-setosa
15	16	5.7	4.4	1.5	0.4	Iris-setosa
16	17	5.4	3.9	1.3	0.4	Iris-setosa
17	18	5.1	3.5	1.4	0.3	Iris-setosa
18	19	5.7	3.8	1.7	0.3	Iris-setosa
19	20	5.1	3.8	1.5	0.3	Iris-setosa
Iris.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 6 columns):
Id               150 non-null int64
SepalLengthCm    150 non-null float64
SepalWidthCm     150 non-null float64
PetalLengthCm    150 non-null float64
PetalWidthCm     150 non-null float64
Species          150 non-null object
dtypes: float64(4), int64(1), object(1)
memory usage: 7.1+ KB
Iris.describe()
Id	SepalLengthCm	SepalWidthCm	PetalLengthCm	PetalWidthCm
count	150.000000	150.000000	150.000000	150.000000	150.000000
mean	75.500000	5.843333	3.054000	3.758667	1.198667
std	43.445368	0.828066	0.433594	1.764420	0.763161
min	1.000000	4.300000	2.000000	1.000000	0.100000
25%	38.250000	5.100000	2.800000	1.600000	0.300000
50%	75.500000	5.800000	3.000000	4.350000	1.300000
75%	112.750000	6.400000	3.300000	5.100000	1.800000
max	150.000000	7.900000	4.400000	6.900000	2.500000
Iris.columns
Index(['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm',
       'Species'],
      dtype='object')
EDA
lets create some simple plots to check out the data
Iris['Species'].value_counts()  #specifies the species and the number
Iris-setosa        50
Iris-virginica     50
Iris-versicolor    50
Name: Species, dtype: int64
sns.countplot(Iris['Species'],label="Count")
plt.show()

​
​
Iris.drop('Id',inplace=True,axis=1)  #to drop the unwanted column i.e 1st column
Iris.head()
SepalLengthCm	SepalWidthCm	PetalLengthCm	PetalWidthCm	Species
0	5.1	3.5	1.4	0.2	Iris-setosa
1	4.9	3.0	1.4	0.2	Iris-setosa
2	4.7	3.2	1.3	0.2	Iris-setosa
3	4.6	3.1	1.5	0.2	Iris-setosa
4	5.0	3.6	1.4	0.2	Iris-setosa
Iris['Species'].unique()
array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], dtype=object)
j=sns.pairplot(Iris,hue='Species',markers=["o","s","D"])
plt.show()

from pandas.plotting import scatter_matrix
from matplotlib import cm
feature_names=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
x=Iris[feature_names]
y=Iris['Species']
Iris.replace({'Species':{'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}})
cmap=cm.get_cmap('gnuplot')
scatter_matrix(x,alpha=0.5,figsize=(10,10))
plt.show()
                                                                    

from pandas.plotting import scatter_matrix
#conversions of categorical data into numeric data
dict={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
Iris.replace({'Species':dict},inplace=True)
Iris['Species'].unique()
array([0, 1, 2], dtype=int64)
#splitting the dataset into a training set and a testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=5)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
(105, 4)
(105,)
(45, 4)
(45,)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
Perform feature scaling
#K-Nearest Neighbour (KNN) Classifier
​
from sklearn.neighbors import KNeighborsClassifier
model_KNN=KNeighborsClassifier().fit(x_train,y_train)
print('Accuracy of K-Nearest Neighbor Classifier is {:.2f}'.format(model_KNN.score(x_test,y_test)))
Accuracy of K-Nearest Neighbor Classifier is 0.96
Logistic regression
from sklearn.linear_model import LogisticRegression
model_lr=LogisticRegression().fit(x_train,y_train)
print('Accuracy of Logistic Regression is{:.2f}'.format(model_lr.score(x_test,y_test)))
​
Accuracy of Logistic Regression is0.89
Random Forest Classifier
from sklearn.svm import SVC
model_svm=SVC().fit(x_train,y_train)
print("Accuracy of support Vector Machine Classifier is {:.2f}".format(model_svm.score(x_test,y_test)))
Accuracy of support Vector Machine Classifier is 0.96
Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model_tree=DecisionTreeClassifier().fit(x_train,y_train)
print("Accuracy of decision tree Classifier is {:.2f}".format(model_tree.score(x_test,y_test)))
Accuracy of decision tree Classifier is 0.91
from sklearn.ensemble import RandomForestClassifier
model_ensemble=RandomForestClassifier().fit(x_train,y_train)
print("Accuracy of random forest Classifier is {:.2f}".format(model_ensemble.score(x_test,y_test)))
Accuracy of random forest Classifier is 0.96
from sklearn.naive_bayes import GaussianNB
model_naive_bayes=GaussianNB().fit(x_train,y_train)
print("Accuracy of gaussianNB Classifier is {:.2f}".format(model_naive_bayes.score(x_test,y_test)))
Accuracy of naive byes Classifier is 0.93
from sklearn.ensemble import AdaBoostClassifier
model_ensemble=AdaBoostClassifier().fit(x_train,y_train)
print("Accuracy of adaboost Classifier is {:.2f}".format(model_ensemble.score(x_test,y_test)))
Accuracy of adaboost Classifier is 0.93
​
