#A cloth manufacturing company is interested to know about the segment or 
#attributes causes high sale. 
#Approach - A Random Forest can be built with target variable Sale
# (we will first convert it in categorical variable) & 
#all other variable will be independent in the analysis.  

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#loading dataset Company_dataset
data = pd.read_csv(r"filepath\Company_Data.csv")

#Data Preprocessing and EDA
data.head()
len(data['Sales'].unique())
data.isnull().sum()
colnames = list(data.columns)
predictors = colnames[1:11]
target = colnames[0]
data["Sales"].max()
data["Sales"].min()

#new dataframe created for preprocessing
data_new=data.copy()

#Making categorical data for Target column Sales
# 1 : Sales<=5
# 2 :5>Sales<=10
# 3 : Sales>10
for i in range(0,len(data)):
    if(data_new["Sales"][i]<=5):
        data_new["Sales"][i]="<=5"
    elif(data_new["Sales"][i]<=10 and data_new["Sales"][i]>5):
        data_new["Sales"][i]="5>s<=10"
    else:    
        data_new["Sales"][i]=">10"

data_new.Sales.value_counts()
#Mapping columns which are categorical to dummy variables
data_new.ShelveLoc=data_new.ShelveLoc.map({"Bad":1,"Good":3,"Medium":2})        
data_new.Urban=data_new.Urban.map({"Yes":1,"No":2})
data_new.US=data_new.US.map({"Yes":1,"No":2})        
data_new.Sales=data_new.Sales.map({"<=5":1,"5>s<=10":2,">10":3})


# Splitting data into training and testing data set by 70:30  ratio

from sklearn.model_selection import train_test_split
train,test = train_test_split(data_new,test_size = 0.3)

from sklearn.ensemble import RandomForestClassifier
rfsales = RandomForestClassifier(n_jobs=4,oob_score=True,n_estimators=30,criterion="entropy")


from sklearn.metrics import confusion_matrix

rfsales.fit(train[predictors],train[target])
# Training Accuracy
train["rf_pred"] = rfsales.predict(train[predictors])
confusion_matrix(train[target],train["rf_pred"]) # Confusion matrix
# Accuracy
print ("Accuracy",(58+169+52)/(58+169+52+1)) # 0.996

test["rf_pred"] = rfsales.predict(test[predictors])
confusion_matrix(test[target],test["rf_pred"])
# Accuracy 
print ("Accuracy",(2+72+8)/(2+72+8+17+1+3+17))#68.33



