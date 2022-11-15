import pandas as pd
import numpy as np


df = pd.read_csv(r'data.csv')

# one column has 1470 data entries
# print(df[df.columns[1]]) # accessing values based on column

allColumnsNames = np.array(df.columns)

# predicting job satisfaction
JSat = df[allColumnsNames[17]]
df.drop(allColumnsNames[17], axis=1) # drop job satisfacftion column from input data

# converting everything to floats...
for columnName in allColumnsNames:
    if columnName == allColumnsNames[1]:
        df[columnName] = df[columnName].replace(to_replace="Yes", value=1, regex=True)
        df[columnName] = df[columnName].replace(to_replace="No", value=0, regex=True)
    if columnName == allColumnsNames[2]:
        df[columnName] = df[columnName].replace(to_replace="Non-Travel", value=0, regex=True)
        df[columnName] = df[columnName].replace(to_replace="Travel_Rarely", value=1, regex=True)
        df[columnName] = df[columnName].replace(to_replace="Travel_Frequently", value=2, regex=True)
    if columnName == allColumnsNames[4]:
        df[columnName] = df[columnName].replace(to_replace="Sales", value=0, regex=True)
        df[columnName] = df[columnName].replace(to_replace="Research & Development", value=1, regex=True)
        df[columnName] = df[columnName].replace(to_replace="Human Resources", value=2, regex=True)
    if columnName == allColumnsNames[7]:
        df[columnName] = df[columnName].replace(to_replace="Life Sciences", value=0, regex=True)
        df[columnName] = df[columnName].replace(to_replace="Medical", value=1, regex=True)
        df[columnName] = df[columnName].replace(to_replace="Technical Degree", value=2, regex=True)
        df[columnName] = df[columnName].replace(to_replace="Marketing", value=3, regex=True)
        df[columnName] = df[columnName].replace(to_replace="Human Resources", value=4, regex=True)
        df[columnName] = df[columnName].replace(to_replace="Other", value=5, regex=True)
    if columnName == allColumnsNames[11]:
        df[columnName] = df[columnName].replace(to_replace="Male", value=1, regex=True)
        df[columnName] = df[columnName].replace(to_replace="Female", value=0, regex=True)
    if columnName == allColumnsNames[15]:
        df[columnName] = df[columnName].replace(to_replace="Life Sciences", value=0, regex=True)
        df[columnName] = df[columnName].replace(to_replace="Medical", value=1, regex=True)
        df[columnName] = df[columnName].replace(to_replace="Technical Degree", value=2, regex=True)
        df[columnName] = df[columnName].replace(to_replace="Marketing", value=3, regex=True)
        df[columnName] = df[columnName].replace(to_replace="Human Resources", value=4, regex=True)
        df[columnName] = df[columnName].replace(to_replace="Other", value=5, regex=True)
    

    
## --------------------- viewing columns by index ----------------------- ##
# count = 0
# for i in allColumnsNames:
#     count += 1
#     print(count, ": ", i)
## -----------------------------------------------------------------------##
# 0 :  Age 
# 1 :  Attrition
# 2 :  BusinessTravel
# 3 :  DailyRate
# 4 :  Department
# 5 :  DistanceFromHome
# 6 :  Education
# 7 :  EducationField
# 8 :  EmployeeCount
# 9 :  EmployeeNumber
# 10 :  EnvironmentSatisfaction
# 11 :  Gender
# 12 :  HourlyRate
# 13 :  JobInvolvement
# 14 :  JobLevel
# 15 :  JobRole
# 16 :  JobSatisfaction
# 17 :  MaritalStatus
# 18 :  MonthlyIncome
# 19 :  MonthlyRate
# 20 :  NumCompaniesWorked 
# 21 :  Over18
# 22 :  OverTime
# 23 :  PercentSalaryHike
# 24 :  PerformanceRating
# 25 :  RelationshipSatisfaction
# 26 :  StandardHours
# 27 :  StockOptionLevel
# 28 :  TotalWorkingYears
# 29 :  TrainingTimesLastYear
# 30 :  WorkLifeBalance
# 31 :  YearsAtCompany
# 32 :  YearsInCurrentRole
# 33 :  YearsSinceLastPromotion
# 34 :  YearsWithCurrManager
## -----------------------------------------------------------------------##

tunableColumnsNames = np.array([allColumnsNames[2],allColumnsNames[3],allColumnsNames[12],allColumnsNames[14],allColumnsNames[18],allColumnsNames[19],allColumnsNames[26],allColumnsNames[27],allColumnsNames[29],allColumnsNames[33],allColumnsNames[34]])
# print(tunableColumnsNames)


# MLP
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
# classifier.fit(df,JSat)
# print(allColumnsNames)