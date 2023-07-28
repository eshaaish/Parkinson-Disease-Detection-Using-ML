"""proposed algorithm:
The data for Parkinson Disease patients and healthy people are collected from various University,
who performed the studies
● The data is converted to structured format i.e. taking the final and initial reading, maximum,
minimum, mean, skewness and median values from the dataset and adding these values to a csv file
to make a proper dataset.
● The algorithms most suitable for detection of PD are found using the K-fold cross validation method.
● The algorithms most suitable found out are XGBoost, Support Vector Machines (SVM), Random
Forests and K-Nearest Neighbour (kNN).
● Each algorithm has its own advantages for training a model and detecting diseases.
● The SVM model plots the healthy and diseased patients on a graph and tries to differentiate and
classify the two types into different spaces.
● It tries to draw a line/space between the two and detects the new patient depending upon which side
of the line they are on.
● The Random Forest algorithm takes various sets of same samples and trains all the sets using
Decision Trees and takes a majority vote of the trees to determine the final result of detection.
● The K-NN algorithm assumes the similarity between the new case/data and available cases and puts
the new case into the category that is most similar to the available categories.
● K-NN algorithm stores all the available data and classifies a new data point based on the similarity.
● XGBoost is a supervised learning algorithm that attempts to accurately predict a target variable by
combining an ensemble of estimates from a set of simpler and weaker models.
● All these algorithms are trained and tested from the dataset made and accuracy for each of the
algorithms is found out.
● The algorithm with the best accuracy is taken as the best algorithm for detecting Parkinson’s disease.
● This algorithm is then used to predict a new set of data and check if the patient has Parkinson Disease
or not."""
Knn Algorithm:
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model 
dfread= pd.read_csv('UCI_Dataset_on_Voice.csv')
#print(dfread)

df = pd.DataFrame(dfread)

dfHealthy =  df[df.status==0]
dfDetected = df[df.status==1]


X = df.drop(['name', 'status'], axis = 'columns')
#print(X.head())
Y = df.status
#print(Y.head())
accuracy =[]
i_value =[]

print(X)
from sklearn.model_selection import train_test_split
acc_avg=0
from sklearn.neighbors import KNeighborsClassifier
for i in range(1,101):
    X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size=0.005)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, Y_train)
    i_value.append(i)
    acc_avg = acc_avg + model.score(X_test, Y_test)
    accuracy.append(model.score(X_test, Y_test))
plt.plot(i_value, accuracy)
print(acc_avg/100)
print(X_test)
sample_test = [119.992, 157.302,74.997,0.00784,0.00007,0.0037,0.00554,0.01109,0.04374,0.426,0.02182,0.0313,0.02971,0.06545,0.02211,21.033,2.301442,0.414783,0.815285,-4.813031,0.266482,0.284654]
print(sample_test)
df4 = pd.DataFrame(sample_test)
print(df4)
#print(model.predict(sample_test))               
Random Forest Algorithm:
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from sklearn import linear_model 
dfread= pd.read_csv('UCI_Dataset_on_Voice.csv')
#print(dfread)

df = pd.DataFrame(dfread)

dfHealthy =  df[df.status==0]
dfDetected = df[df.status==1]


X = df.drop(['name', 'status'], axis = 'columns')
#print(X.head())
Y = df.status
#print(Y.head())
accuracy =[]
i_value =[]

from sklearn.model_selection import train_test_split
acc_avg=0
from sklearn.ensemble import RandomForestClassifier
for i in range(1,101):
    X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size=0.5)
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)
    i_value.append(i)
    acc_avg = acc_avg + model.score(X_test, Y_test)
    accuracy.append(model.score(X_test, Y_test))
plt.plot(i_value, accuracy)
print(acc_avg/100)

SVM algorithm:
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.svm import SVC  
dfread= pd.read_csv('UCI_Dataset_on_Voice.csv')
#print(dfread)

df = pd.DataFrame(dfread)

dfHealthy =  df[df.status==0]
dfDetected = df[df.status==1]

#plt.scatter(dfHealthy['spread1'], dfHealthy['spread2'], color='green')
#plt.scatter(dfDetected['spread1'], dfDetected['spread2'], color='blue')

X = df.drop(['name', 'status'], axis = 'columns')
#print(X.head())
Y = df.status
#print(Y.head())
accuracy =[]
c_value =[]

for i in range(1,100):
    X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size=0.2)
    model=SVC(C=i,gamma=i)
    model.fit(X_train, Y_train)
    c_value.append(i)
    accuracy.append(model.score(X_test, Y_test))
plt.plot(c_value, accuracy)
plt.xlabel("C and gamma variation")
plt.ylabel("Accuracy")

dfpredict = pd.read_csv('Testing_values.csv')
#print(dfpredict)
dfpredict = dfpredict.drop(['name'], axis='columns')

print(model.predict(dfpredict))


XGBoost Algorithm:
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model 
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split

dfread= pd.read_csv('UCI_Dataset_on_Voice.csv')
#print(dfread)

df = pd.DataFrame(dfread)


X = df.drop(['name', 'status'], axis = 'columns')
#print(X.head())
#scalar=MinMaxScalar((-1,1))
#X = scalar.fit_transform(X)
Y = df.status
#print(Y.head())
accuracy =[]
i_value =[]
acc_avg=0

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)
for i in range(1, 2):   
    X_train, X_test, Y_train, Y_test  = train_test_split(X_scaled, Y, test_size=0.2)
    xg_reg = xgb.XGBClassifier(eval_metric = 'mlogloss', use_label_encoder=False)
    xg_reg.fit(X_train, Y_train)
    accuracy.append(xg_reg.score(X_test, Y_test))
    acc_avg = acc_avg + xg_reg.score(X_test, Y_test)
    i_value.append(i)
#plt.plot(i_value, accuracy)
#print(acc_avg/100)
print(X_test)


#implementing using Bagging along with XGBoost
bag_i=[]
bag_acc=[]
bag_accu=0
from sklearn.ensemble import BaggingClassifier
#for i in range(1, 31):
#    bag_model = BaggingClassifier(base_estimator= xgb.XGBClassifier(eval_metric = 'mlogloss', use_label_encoder=False), n_estimators=30, oob_score = True)
#    bag_model.fit(X_train, Y_train)
#    bag_i.append(i)
#    score = bag_model.score(X_test, Y_test)
#    bag_acc.append(score)
#    bag_accu = bag_accu + score
    #print(score)
    
#plt.plot(bag_i, bag_acc, color="red")
#print("After baggin: ", bag_accu/30)
