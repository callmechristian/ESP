import pandas as pd
import numpy as np
import os

df = pd.read_csv(r'data/data_processed.csv')
df_continous = pd.read_csv(r'data/data_continous.csv')
df_categorical = pd.read_csv(r'data/data_categorical.csv')

# --------------------------------------------------------------------------------------------------
# copying job satisfaction into new array
JSat = df['JobSatisfaction'].values

# splitting inputs by row index
# continous data
df_training_continous = df_continous.iloc[:1400,:]
df_validation_continous = df_continous.iloc[1400:,:]
# categorical data
df_training_categorical = df_categorical.iloc[:1400,:]
df_validation_categorical = df_categorical.iloc[1400:,:]
# splitting outputs by number
JSat_training = JSat[:1400]
JSat_validation = JSat[1400:]
# --------------------------------------------------------------------------------------------------
# tunableColumnsNames = np.array([allColumnsNames[2],allColumnsNames[3],allColumnsNames[12],allColumnsNames[14],allColumnsNames[18],allColumnsNames[19],allColumnsNames[26],allColumnsNames[27],allColumnsNames[29],allColumnsNames[33],allColumnsNames[34]])
# print(tunableColumnsNames)

# MLP
# from sklearn.neural_network import MLPClassifier
# from sklearn import metrics
# classifier = Random
# classifier.fit(df_training,JSat_training)
# JSat_predicted = classifier.predict(df_validation)
# print(metrics.classification_report(JSat_validation,JSat_predicted))



# Random Forests
from sklearn.ensemble import RandomForestClassifier

# continous data classifier
clf_continous = RandomForestClassifier(max_depth=None, random_state=None)

clf_continous.fit(df_training_continous,JSat_training)
acc_continous = clf_continous.score(df_validation_continous, JSat_validation)

# categorical data classifier
clf_categorical = RandomForestClassifier(max_depth=None, random_state=None)

clf_categorical.fit(df_training_categorical,JSat_training)
acc_categorical = clf_categorical.score(df_validation_categorical, JSat_validation)

# print results
print('Continous data accuracy: ', acc_continous)
print('Categorical data accuracy: ', acc_categorical)

# print(allColumnsNames)