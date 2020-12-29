#Use RandomForest to prepare a model on fraud data 
#treating those who have taxable_income <= 30000 as "Risky" and others are "Good"
#Data Description :
#Undergrad : person is under graduated or not
#Marital.Status : marital status of a person
#Taxable.Income : Taxable income is the amount of how much tax an individual owes to the government 
#Work Experience : Work experience of an individual person
#Urban : Whether that person belongs to urban area or not

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#loading dataset Company_dataset
data = pd.read_csv(r"filepath\Fraud_check.csv")

#Data Preprocessing and EDA
data.head()
data.columns
len(data['Taxable.Income'].unique())
data.isnull().sum()
colnames= list(data.columns)
target=colnames[2]
colnames.pop(2)
predictors=colnames

#new dataframe created for preprocessing
data_new=data.copy()

#Making categorical data for Target column Taxable.Income
# 1 : Taxable.Income<=30000 Risky
# 2 : Taxable.Income>30000 Good
for i in range(0,len(data_new)):
    if(data_new["Taxable.Income"][i]<=30000):
        data_new["Taxable.Income"][i]="Risky"
    else:    
        data_new["Taxable.Income"][i]="Good"
        
data_new["Taxable.Income"].value_counts()        

#Mapping columns which are categorical to dummy variables
data_new["Taxable.Income"]=data_new["Taxable.Income"].map({"Risky":1,"Good":2})
data_new["Undergrad"]=data_new["Undergrad"].map({"YES":1,"NO":2}) 
data_new["Marital.Status"]=data_new["Marital.Status"].map({"Single":1,"Divorced":2,"Married":3})
data_new["Urban"]=data_new["Urban"].map({ "YES":1,"NO":2})


# Splitting data into training and testing data set by 70:30 ratio

from sklearn.model_selection import train_test_split
train,test = train_test_split(data_new,test_size = 0.3)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")
rf.fit(train[predictors],train[target])

preds = rf.predict(test[predictors])
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)

# Accuracy = train  
np.mean(train["Taxable.Income"] == rf.predict(train[predictors]))#0.9857

# Accuracy = Test
np.mean(preds==test["Taxable.Income"]) # 0.7055

