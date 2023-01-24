import pandas as pd
import numpy as np
import os

df = pd.read_csv(r'data/data_processed.csv')
df_continous = pd.read_csv(r'data/data_continous.csv')
df_categorical = pd.read_csv(r'data/data_categorical.csv')

# --------------------------------------------------------------------------------------------------
# copying job satisfaction into new array
JSat = df['JobSatisfaction'].values

df.drop(['JobSatisfaction'])

# splitting inputs by row index
# continous data
df_training_continous = df_continous.iloc[:1200,:]
df_validation_continous = df_continous.iloc[1200:,:]
# categorical data
df_training_categorical = df_categorical.iloc[:1200,:]
df_validation_categorical = df_categorical.iloc[1200:,:]
# splitting outputs by number
JSat_training = JSat[:1200]
JSat_validation = JSat[1200:]
# --------------------------------------------------------------------------------------------------

# Random Forests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# categorical data classifier
clf_categorical = RandomForestClassifier(max_depth=None, random_state=None)

clf_categorical.fit(df_training_categorical,JSat_training)
acc_categorical = clf_categorical.score(df_validation_categorical, JSat_validation)

pred = clf_categorical.predict(df_validation_categorical)

print(classification_report(JSat_validation, pred))

# print results
print('Categorical data accuracy: ', acc_categorical)

# print(allColumnsNames)