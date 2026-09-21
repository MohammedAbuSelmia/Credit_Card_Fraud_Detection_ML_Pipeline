import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
##- First, I want to send the file to the SQL database to make the work more organized and professional.
data=pd.read_csv(r'creditcard.csv') # i extract data from csv file
print(data.head(5))
server_name='DESKTOP-S0OHOAK'
database_name='Credit-Card-Fraud-Detection-ML-Pipeline' 
en=create_engine(f'mssql+pyodbc://{server_name}/{database_name}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes') #creat engine to access to database
data.to_sql(name=database_name,chunksize=100,index=False,if_exists='replace',con=en) # in this code i sent data from csv to database in  SSMS  
print('Data transfer was successful') # This code mean the transfer process was successful. 
qu="""
select * from [Credit-Card-Fraud-Detection-ML-Pipeline]
"""
data=pd.read_sql(qu,con=en) #now i want to read the data from sql file in ssms
##- Explore data
print(data.sample(10)) 
print(data.info())
print(data.isna().sum())# in this code i Check if there is nan value or not

print(data.corr()) # in this i explor who featuers linked together
##- Here we notice something very important: the percentage of fraudulent activity was very low, and this is a very important indicator when we want to build a prediction model.
print(((data.loc[data['Class']==1,['Class']].sum())/(data['Class'].count()))*100)


##- ok now after check the data and once i understand it and know everything about it, i move on to the transform phase
##- First, I noticed something very important: the time column is written in seconds. I want to convert it to hours so I can analyze it better

data['Time']=round(((data['Time']/3600)%24).astype(float),4)
d=data.loc[data['Class']==1,'Class']
data['Time']=data['Time'].astype(int)# In this code i changed the data type to int to identify the hours when fraud occurs most frequently.
fraud_data = data[data['Class'] == 1] 
print(fraud_data.sample(10))
#now i want to focus on case fraud
print(fraud_data['Time'].value_counts()) # After print this code We note something very important: the hours during which most scams occurred
plt.figure(figsize=(12, 6))

##- Now I want to draw the case fraud to be more clear
sns.histplot(fraud_data['Time'],color='r',bins=24,kde=True)
plt.title('Distribution of fraud based on the hour', fontsize=15)
plt.xlabel('Hour', fontsize=12)
plt.ylabel('Number of scams', fontsize=12)
plt.xticks(range(0, 24))
plt.show()

##-Now i want to see the details amount of stolen money
print(fraud_data['Amount'].aggregate(['mean','max','median','min','sum']))# in this code show the mean amount and max and median and  min and total amount stolen
## لاحظت انه تم سحب مبالغ بقيمة صفر وهذا شئ مهم الان اريد ان ارى كم نسبت المبالغ التي تم سحبها بقيمة صفر 
print(((fraud_data.loc[fraud_data['Amount']==0,'Amount'].count())/(fraud_data['Amount'].count()))*100)# in this code show the percentage of 0 amount of total
##- Now i want to know the time when money has been withdrawn greater than the mean amount
above_of_mean=fraud_data.loc[fraud_data['Amount']>np.mean(fraud_data['Amount'])]# in this code i extract the transactions have value above then mean
print(above_of_mean['Time'].value_counts())# in this code i showed how many times did a fraud happen every hour

###- Now i want to draw it
sns.histplot(above_of_mean['Time'],color='r',bins=24,kde=True)
plt.title('How many times has an fraud happenned')
plt.xlabel('Hour')
plt.ylabel('Number of fraud')
plt.xticks(range(0,24))
plt.show()

# In the end i want to transform datatype time from int to float to algorithem make better
data['Time']=data['Time'].astype(float)
#############################################################################################################
##-In this section i want to bulid the model to expect willbe an operation fraud or not
# First i want to call the models that i want to work on from the library sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
# because the data not balance i want to be balace to the model moer professional and correct i will use 
from imblearn.over_sampling import RandomOverSampler # i import this class RandomOverSampler from model over_sample to raise number class have 1 to be data more balance 

from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import zero_one_loss

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

##- Now i want to divide data to X and y
X=data.drop('Class',axis=1)
y=data['Class']


##- In this phase i want to cheuck we donot have nan value and if we have i want to replace to mean value
Imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
X=Imputer.fit_transform(X)
# Now I defind varible
standModel=StandardScaler(copy=True,with_mean=True,with_std=True)

##- Now i want to split x and y to be x_train ,x_test ,y_train ,y_test
X_train,X_test,y_tain,y_test=train_test_split(X,y,test_size=.27,random_state=33,shuffle=True)

##- Now I want to be all value in  features  close to each other so taht  the model is more accurate
X_train=standModel.fit_transform(X_train)
X_test=standModel.transform(X_test)

##- Now i want to over fiting data to be more balance
over_sample=RandomOverSampler(random_state=33)
X_train_re,y_tain_re=over_sample.fit_resample(X_train,y_tain)
print(X_train.shape)# to make sure the over fiting happened

##-Now i want to define the algorithms 
logistic_model=LogisticRegression(random_state=33,solver='sag')
forest_model=RandomForestClassifier(random_state=33,n_estimators=100) 

def trans(model):
   
    model.fit(X_train_re,y_tain_re)
    y_pred=model.predict(X_test)
    recall=recall_score(y_test,y_pred)
    zero_one=zero_one_loss(y_test,y_pred,normalize=True)
    cnfusion=confusion_matrix(y_test,y_pred)
    print(f'Name model is => {model} ')
    print(f'The recall is => {recall} ||  and the zero_one_loss is => {zero_one} || and confusion is => {cnfusion}')

trans(logistic_model)
trans(forest_model)



##### In the end 
# Evaluation of the Logistic Regression Model
# Recall (0.907): This is an excellent result. The model correctly identified 128 out of 141 fraud cases (128 + 13), indicating its high level of vigilance and minimal misses of potential crimes.
# Weaknesses: The False Positives are high (1971 legitimate cases were falsely accused of fraud). In the banking industry, this can be inconvenient for some clients, but it is sometimes acceptable to ensure funds are not lost.

# 2. Evaluation of the Random Forest Model
# Overall Accuracy: The Zero-one loss is very low (0.0005), and the False Positives are almost nonexistent (only 5 false positives). This suggests the model is relatively quiet and doesn't generate false alarms.
# Recall (0.709): This is where the problem lies; the model missed 41 fraud cases (False Negatives), a significant number compared to the first model.


##-Now i want to save the model LogisticRegression because the recall was 90% and have a good confusion_matrix
import joblib as j
j.dump(logistic_model,'The_Best_Model_prediction_fraud_model.pkl')
j.dump(standModel, 'Scaler_Transform.pkl')
print('save')
