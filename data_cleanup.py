import pandas as pd
import os

df = pd.read_csv(r'data/data_processed.csv')
cNames = df.columns.values
# print(cNames)

# for i in range(1,cNames.size):
#     print(i, cNames[i])

# split categorical and continuous data
df_continous = df[['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']].copy()
df_categorical = df[['Attrition', 'BusinessTravel', 'Department', 'Education','EducationField','EnvironmentSatisfaction','Gender','HourlyRate','JobInvolvement','JobLevel','JobRole','JobSatisfaction','MaritalStatus','NumCompaniesWorked', 'OverTime','PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel','TrainingTimesLastYear', 'WorkLifeBalance']].copy()

# create data dir if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")
# save data to csv
df_categorical.to_csv("data/data_categorical.csv")
df_continous.to_csv("data/data_continous.csv")
