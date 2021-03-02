# Linear Regression

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error,r2_score

Glance=10
view=mydata.head(Glance)
print (view)

dim=mydata.shape
print(dim)

Data_type=mydata.dtypes
print(Data_type)

#Decription
description=mydata.describe()
print(description)

#Data Defining
column_names =['xg_z','crosses_z','boxtouches_z','takeons_z','progruns_z','clearances_z','aerials_z','fouls_z','fouled_z','nsxg_z']
x=mydata[column_names]
y=mydata[['goals_z']]

#Data frame
df = pd.DataFrame(mydata, columns = ['xg_z','crosses_z','goals_z','boxtouches_z','takeons_z','progruns_z','clearances_z','aerials_z','fouls_z','fouled_z','nsxg_z'])
x=np.array(df[['xg_z','crosses_z','boxtouches_z','takeons_z','progruns_z','clearances_z','aerials_z','fouls_z','fouled_z','nsxg_z']])
y=np.array(df['goals_z'])

#Splitting Data into 70:30 ratio
trainf,testf,trainl,testl= train_test_split(x,y,test_size = .30,random_state=15)
#printing train and test length 
print(len(trainf))
print(len(trainl))
print(len(testf))
print(len(testl))

#Linear Regression
Obj = LinearRegression()

#FITTING THE MODEL
Obj.fit(trainf,trainl)

#PREDICTION
result = Obj.predict(testf)

df = pd.DataFrame({'Actual': testl, 'Predicted': result})
df1 = df.head(10)
df1

