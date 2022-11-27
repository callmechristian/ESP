import pandas as pd
import numpy as np
import os

df = pd.read_csv(r'data/data_processed.csv')

allColumnsNames = np.array(df.columns)

# copying job satisfaction into new array
JSat = df[allColumnsNames[17]].values
# drop job satisfacftion column from input data
df.drop(allColumnsNames[17], axis=1)

# splitting inputs by row index
df_training = df.iloc[:1400,:]
df_validation = df.iloc[1400:,:]
# splitting outputs by number
JSat_training = JSat[:1400]
JSat_validation = JSat[1400:]
print("Shape of new inputs - {} , {}".format(df_training.shape, df_validation.shape))
print("Shape of new outputs - {} , {}".format(JSat_training.shape, JSat_validation.shape))

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
from sklearn.datasets import make_classification

# df_training, JSat_training = make_classification(n_estimators=500, bootstrap=False)
clf = RandomForestClassifier(max_depth=None, random_state=None)
clf.fit(df_training,JSat_training)
JSat_predicted = clf.predict(df_validation)
print(clf.score(df_validation, JSat_validation)) 

# print(allColumnsNames)