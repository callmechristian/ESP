import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'data/data_processed.csv')

# let's inspect our data
# https://towardsdatascience.com/statistics-in-python-collinearity-and-multicollinearity-4cc4dcd82b3f

# vif
from sklearn.linear_model import LinearRegression
def calculate_vif(df, features):    
    vif, tolerance = {}, {}
    # all the features that you want to examine
    for feature in features:
        # extract all the other features you will regress against
        X = [f for f in features if f != feature]        
        X, y = df[X], df[feature]
        # extract r-squared from the fit
        r2 = LinearRegression().fit(X, y).score(X, y)                
        
        # calculate tolerance
        tolerance[feature] = 1 - r2
        # calculate VIF
        vif[feature] = 1/(tolerance[feature])
    # return VIF DataFrame
    return pd.DataFrame({'VIF': vif, 'Tolerance': tolerance})
# relationships between various columns


corr = df.corr()

c = 15
print("                                ",corr.columns.values[c])
print(corr[corr.columns.values[c]].to_string(index=True))

# sns.pairplot(df[df.columns.values[1:10]])
# plt.show()

# create data dir if it doesn't exist
if not os.path.exists("out"):
    os.makedirs("out")
# save corr data to csv
corr.to_csv("out/correlation.csv")