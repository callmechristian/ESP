import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'data/data_processed.csv')
print(len(df.columns))
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
# relationships between various columns%


corr = df.corr()

c = 15
print("                                ",corr.columns.values[c])
print(corr[corr.columns.values[c]].to_string(index=True))


# split categorical and continuous data
# df_continous = df[['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']].copy()
# df_categorical = df[['Attrition', 'BusinessTravel', 'Department', 'Education','EducationField','EnvironmentSatisfaction','Gender','HourlyRate','JobInvolvement','JobLevel','JobRole','JobSatisfaction','MaritalStatus','NumCompaniesWorked', 'OverTime','PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel','TrainingTimesLastYear', 'WorkLifeBalance']].copy()
# split categorical and continuous data
df_continous = df[['Age', 'DistanceFromHome', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']].copy()
df_categorical = df[['Attrition', 'BusinessTravel', 'Education','EnvironmentSatisfaction','JobInvolvement','JobLevel','JobSatisfaction','MaritalStatus', 'OverTime','PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance']].copy()

df_corr_targets = df[['JobSatisfaction', 'MonthlyIncome', 'DistanceFromHome','Attrition','EnvironmentSatisfaction','JobInvolvement','RelationshipSatisfaction','WorkLifeBalance']].copy()
corr = df_corr_targets.corr()
# sns.pairplot(df_continous[:100])
# for i in range(1,len(corr)):
#     plt.plot(corr[corr.columns.values[i]])
#     plt.title(corr.columns.values[i])
#     plt.xticks(rotation=45, ha='right')
    # plt.show()
plt.plot(corr[corr.columns.values[0]])
plt.title(corr.columns.values[0])
plt.xticks(rotation=45, ha='right')
plt.show()

# sns.pairplot(df[df.columns.values[1:10]])
# plt.show()

# myplot1 = sns.pairplot(df_continous)
# plt.figure()
# myplot2 = sns.pairplot(df_categorical[df_categorical.columns.values[1:5]])
# plt.figure()

# plt.show()

# create data dir if it doesn't exist
if not os.path.exists("out"):
    os.makedirs("out")
# save corr data to csv
corr.to_csv("out/correlation.csv")