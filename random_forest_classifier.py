import pandas as pd
import numpy as np
import os

df = pd.read_csv(r'data/data_processed.csv')
df_continous = pd.read_csv(r'data/data_continous.csv')
df_categorical = pd.read_csv(r'data/data_categorical.csv')

# --------------------------------------------------------------------------------------------------
# copying job satisfaction into new array
JAtt = df['Attrition'].values

df_good = pd.concat([df_categorical, df["MonthlyIncome"], df["BusinessTravel"], df["StockOptionLevel"], df["DistanceFromHome"]], axis=1)
df = df_good.drop(['Attrition'], axis=1)

# splitting inputs by row index
# all data
df_training = df.iloc[:1200,:]
df_validation = df.iloc[1200:,:]
# continous data
df_training_continous = df_continous.iloc[:1200,:]
df_validation_continous = df_continous.iloc[1200:,:]
# categorical data
df_training_categorical = df_categorical.iloc[:1200,:]
df_validation_categorical = df_categorical.iloc[1200:,:]
# splitting outputs by number
JAtt_training = JAtt[:1200]
JAtt_validation = JAtt[1200:]
# --------------------------------------------------------------------------------------------------

# Random Forests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# data classifier
clf = RandomForestClassifier()

clf.fit(df_training,JAtt_training)
acc = clf.score(df_validation, JAtt_validation)
acc_all = clf.score(df, JAtt)

pred = clf.predict(df_validation)
pred_all = clf.predict(df)

print(classification_report(JAtt_validation, pred))
print(classification_report(JAtt, pred_all))

# print results
print('Data accuracy: ', acc)
print('Data accuracy on all samples:', acc_all)

# print(allColumnsNames)