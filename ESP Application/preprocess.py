import pandas as pd
import numpy as np
import os


def clean(df, OP):
    allColumnsNames = df.columns.values
    # converting everything to floats...
    for columnName in allColumnsNames:
        if columnName == allColumnsNames[1]: # Attrition
            df[columnName] = df[columnName].replace(to_replace="Yes", value=1, regex=True)
            df[columnName] = df[columnName].replace(to_replace="No", value=0, regex=True)
        if columnName == allColumnsNames[2]: # Business Travel
            df[columnName] = df[columnName].replace(to_replace="Non-Travel", value=0, regex=True)
            df[columnName] = df[columnName].replace(to_replace="Travel_Rarely", value=1, regex=True)
            df[columnName] = df[columnName].replace(to_replace="Travel_Frequently", value=2, regex=True)
        if columnName == allColumnsNames[4]: # Department
            df[columnName] = df[columnName].replace(to_replace="Sales", value=0, regex=True)
            df[columnName] = df[columnName].replace(to_replace="Research & Development", value=1, regex=True)
            df[columnName] = df[columnName].replace(to_replace="Human Resources", value=2, regex=True)
        if columnName == allColumnsNames[7]: # Education
            df[columnName] = df[columnName].replace(to_replace="Life Sciences", value=0, regex=True)
            df[columnName] = df[columnName].replace(to_replace="Medical", value=1, regex=True)
            df[columnName] = df[columnName].replace(to_replace="Technical Degree", value=2, regex=True)
            df[columnName] = df[columnName].replace(to_replace="Marketing", value=3, regex=True)
            df[columnName] = df[columnName].replace(to_replace="Human Resources", value=4, regex=True)
            df[columnName] = df[columnName].replace(to_replace="Other", value=5, regex=True)
        if columnName == allColumnsNames[11]: # Gender
            df[columnName] = df[columnName].replace(to_replace="Male", value=1, regex=True)
            df[columnName] = df[columnName].replace(to_replace="Female", value=0, regex=True)
        if columnName == allColumnsNames[15]: # Job Role
            df[columnName] = df[columnName].replace(to_replace="Sales Executive", value=0, regex=True)
            df[columnName] = df[columnName].replace(to_replace="Research Scientist", value=1, regex=True)
            df[columnName] = df[columnName].replace(to_replace="Laboratory Technician", value=2, regex=True)
            df[columnName] = df[columnName].replace(to_replace="Manufacturing Director", value=3, regex=True)
            df[columnName] = df[columnName].replace(to_replace="Manager", value=4, regex=True)
            df[columnName] = df[columnName].replace(to_replace="Sales Representative", value=5, regex=True)
            df[columnName] = df[columnName].replace(to_replace="Healthcare Representative", value=6, regex=True)
            df[columnName] = df[columnName].replace(to_replace="Research Director", value=7, regex=True)
            df[columnName] = df[columnName].replace(to_replace="Human Resources", value=8, regex=True)
        if columnName == allColumnsNames[17]: # Marital Status
            df[columnName] = df[columnName].replace(to_replace="Single", value=0, regex=True)
            df[columnName] = df[columnName].replace(to_replace="Married", value=1, regex=True)
            df[columnName] = df[columnName].replace(to_replace="Divorced", value=2, regex=True)
        if columnName == allColumnsNames[21]: # Over 18
            df[columnName] = df[columnName].replace(to_replace="Y", value=1, regex=True)
            df[columnName] = df[columnName].replace(to_replace="N", value=0, regex=True)
        if columnName == allColumnsNames[22]: # OverTime
            df[columnName] = df[columnName].replace(to_replace="Yes", value=1, regex=True)
            df[columnName] = df[columnName].replace(to_replace="No", value=0, regex=True)

    droplist = ['BusinessTravel','DailyRate','EmployeeCount','EmployeeNumber','HourlyRate','MonthlyRate','NumCompaniesWorked','Over18','StandardHours','TrainingTimesLastYear']
    for val in droplist:
        try:
            df.drop(val, axis=1, inplace=True)
        except:
            continue
        # print(val + " dropped from the dataset successfully.")

    dataset = df.reindex(columns = [col for col in df.columns if col != OP] + [OP])
    return dataset

