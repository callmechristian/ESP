import pandas as pd
import numpy as np
import os


df = pd.read_csv(r'data/data.csv')

# one column has 1470 data entries
# print(df[df.columns[1]]) # accessing values based on column
df = df.rename(columns=lambda x: x.strip())
allColumnsNames = df.columns.values

# converting everything to floats...
for columnName in allColumnsNames:
    # print("'",columnName,"'")
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
        df[columnName] = df[columnName].replace(to_replace="Sales Executive", value=0, regex=True)
        df[columnName] = df[columnName].replace(to_replace="Research Scientist", value=1, regex=True)
        df[columnName] = df[columnName].replace(to_replace="Laboratory Technician", value=2, regex=True)
        df[columnName] = df[columnName].replace(to_replace="Manufacturing Director", value=3, regex=True)
        df[columnName] = df[columnName].replace(to_replace="Manager", value=4, regex=True)
        df[columnName] = df[columnName].replace(to_replace="Sales Representative", value=5, regex=True)
        df[columnName] = df[columnName].replace(to_replace="Healthcare Representative", value=6, regex=True)
        df[columnName] = df[columnName].replace(to_replace="Research Director", value=7, regex=True)
        df[columnName] = df[columnName].replace(to_replace="Human Resources", value=8, regex=True)
    if columnName == allColumnsNames[17]:
        df[columnName] = df[columnName].replace(to_replace="Single", value=0, regex=True)
        df[columnName] = df[columnName].replace(to_replace="Married", value=1, regex=True)
        df[columnName] = df[columnName].replace(to_replace="Divorced", value=2, regex=True)
    if columnName == allColumnsNames[21]:
        df[columnName] = df[columnName].replace(to_replace="Y", value=1, regex=True)
        df[columnName] = df[columnName].replace(to_replace="N", value=0, regex=True)
    if columnName == allColumnsNames[22]:
        df[columnName] = df[columnName].replace(to_replace="Yes", value=1, regex=True)
        df[columnName] = df[columnName].replace(to_replace="No", value=0, regex=True)
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
# 1 :  Age
# 2 :  Attrition
# 3 :  BusinessTravel
# 4 :  DailyRate
# 5 :  Department
# 6 :  DistanceFromHome
# 7 :  Education
# 8 :  EducationField
# 9 :  EnvironmentSatisfaction
# 10 :  Gender
# 11 :  HourlyRate
# 12 :  JobInvolvement
# 13 :  JobLevel
# 14 :  JobRole
# 15 :  MaritalStatus
# 16 :  MonthlyIncome
# 17 :  MonthlyRate
# 18 :  NumCompaniesWorked
# 19 :  OverTime
# 20 :  PercentSalaryHike
# 21 :  PerformanceRating
# 22 :  RelationshipSatisfaction
# 23 :  StockOptionLevel
# 24 :  TotalWorkingYears
# 25 :  TrainingTimesLastYear
# 26 :  WorkLifeBalance
# 27 :  YearsAtCompany
# 28 :  YearsInCurrentRole
# 29 :  YearsSinceLastPromotion
# 30 :  YearsWithCurrManager

# remove useless columns
df.drop(allColumnsNames[8], axis=1, inplace=True)
df.drop(allColumnsNames[9], axis=1, inplace=True)
df.drop(allColumnsNames[21], axis=1, inplace=True)
df.drop(allColumnsNames[26], axis=1, inplace=True)

# print(df)
# new columns and index number
count = 0
for i in df.columns.values:
    count += 1
    print(count, ": ", i)

# shuffle data
df = df.sample(frac=1).reset_index(drop=True)

# create data dir if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")
# save beautified data to csv
df.to_csv("data/data_processed.csv")