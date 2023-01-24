import sys
import os
import json
import pickle
import random
import pandas as pd
from preprocess import clean

# Flask utils
from flask import Flask, redirect, url_for, request, render_template

# Define a flask app
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

def predict(data, model, OP, AP):
    js = json.loads(data)
    df = pd.json_normalize(js)
    dataset = clean(df, OP)
    dataset = dataset.drop(AP, axis=1)
    dataset = dataset.drop(OP, axis=1)
    loaded_model = pickle.load(open(model, 'rb'))
    result = loaded_model.predict(dataset)[0]
    return result

@app.route('/attritionapi', methods=['POST'])
def attrition():
    if request.method == 'POST':
        # print(type(request.json))
        data = json.dumps(request.json)
        # print(type(data))
        res = predict(data, "Attrition.sav", "Attrition", "JobSatisfaction")
        print(res)
        msg = ""
        if(res == 0):
            return "The employee won't be leaving anytime soon."
        elif(res == 1):
            x = json.loads(data)
            if(res==1 and x["PerformanceRating"] != 4):
                x["PerformanceRating"] = 4
                data = json.dumps(x)
                msg = "increase their Performance rating to 4"
                print("Performance Rating Updated")
                res = predict(data, "Attrition.sav", "Attrition", "JobSatisfaction")
            if(res==1 and x["EnvironmentSatisfaction"] != 4):
                x["EnvironmentSatisfaction"] = 4
                data = json.dumps(x)
                msg = "change them to a better work environment"
                res = predict(data, "Attrition.sav", "Attrition", "JobSatisfaction")
                print("Environment Satisfaction Updated")
            if(res==1 and x["OverTime"] == "Yes"):
                x["OverTime"] = "No"
                data = json.dumps(x)
                msg = "stop the Over time work"
                res = predict(data, "Attrition.sav", "Attrition", "JobSatisfaction")
            if(res==1 and x["JobLevel"] != 4):
                x["JobLevel"] = 4
                data = json.dumps(x)
                msg = "increase their job level through promotion"
                res = predict(data, "Attrition.sav", "Attrition", "JobSatisfaction")
                print("Job Level Updated")
            if(res==1 and x["PercentSalaryHike"] != 25):
                x["PercentSalaryHike"] = 25
                data = json.dumps(x)
                msg = "increase their salary by 25%"
                res = predict(data, "Attrition.sav", "Attrition", "JobSatisfaction")
                print("Percent Salary Hike Updated")
            if(res == 0):
                msg = "The employee might leave the organisation soon. To retain this employee back, you can "+msg+"!"
            else:
                msg = "The employee might leave the organisation soon!"
            return msg
    return None

@app.route('/satisfactionapi', methods=['POST'])
def jobsatisfaction():
    if request.method == 'POST':
        data = json.dumps(request.json)
        res = predict(data, "JobSatisfaction.sav", "JobSatisfaction", "Attrition")
        print(res)
        return "Job Satisfaction of the employee - "+str(res)+"/5"
    return None

@app.route('/samples', methods=['GET', 'POST'])
def sampledata():
    lister = []
    with open("data.csv" , 'r') as f:
        heading = f.readline()
        for line in f:
            lister.append(line.replace("\n", ""))
    print(len(lister))
    samples = random.sample(lister, 5)
    print(len(samples))
    return render_template('samples.html', samples=samples, len=len, range=range)

if __name__ == '__main__':    
    app.run(debug=True, threaded=False)