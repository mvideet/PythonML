#Import Statements
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.mlab as mlab
#Input
# symptoms =  ['SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE','CHRONIC DISEASE','FATIGUE ',  'WHEEZING', 'ALCOHOL CONSUMING','COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
#
# #age = int(input("What is your age? "))
# #smoking = int(input("Do you smoke?(1 for No and 2 for Yes) "))
# input_arr = []
# for i in range(6):
#     ans = int(input("Do you experience " + str(symptoms[i]) + "?(1 for No and 2 for Yes) "))
#     input_arr.append(ans)
# input = pd.DataFrame([input_arr], columns = ['SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE','CHRONIC DISEASE','FATIGUE ',  'WHEEZING', 'ALCOHOL CONSUMING','COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN'])
#Reading the csv file with all the data
df = pd.read_csv("/Users/mvideet/PycharmProjects/CAC/lungcancer.csv")
df = df.replace({"M":0, "F":1, "NO":0, "YES":1})
#print(df.head())
starting_size = df.size
df.dropna(axis = 0, inplace = True)

symptoms =   ['SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE','CHRONIC DISEASE','FATIGUE ',  'WHEEZING', 'ALCOHOL CONSUMING','COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
#for i in symptoms:
 #   df[i] = df[i].map({2:'Positive', 1:'Negative'})
df.head()
#Creating the dataframe to put into the Random Forest Regressor
x=df[symptoms]
y=df.LUNG_CANCER
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=(0.2), random_state=42)
# forest_regressor = RandomForestClassifier(n_estimators=100, random_state=42)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(x_train),columns=x_train.columns)
X_train_scaled
X_test_scaled = pd.DataFrame(scaler.transform(y_test),columns=y_test.columns)
X_test_scaled
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
ros =  RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X_train_scaled, y_train)
print('Original dataset shape', Counter(y))
print('Resample dataset shape', Counter(y_resampled))
print(sorted(Counter(y_resampled).items()))
from sklearn.ensemble import GradientBoostingClassifier