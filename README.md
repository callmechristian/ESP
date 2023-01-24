# Employee Satisfaction Project - ESP
***

A project for Data Science in Business

### Utilizing the available scripts
The scripts in the main repository were used to identify a model and create statistics and insights for different models. While a little bit out of scope, and almost nothing could be included in the report, I'll give a short summary of what everything does.

1. interpret_* scripts display computed statistics and insights from the generated models (no plots, because there's too many models)
2. classifier_search* scripts implement a custom grid search for 32 or selected models
3. data_* scripts work with the dataset, trim it, encode it, provide statistics about the input data
4. randomized_search_* implement a randomized hyperparameter search for the specific models
5. tf_keras_* implement a neural neural network model, but it was more expensive and provided worse results than simple algorithms

random_forest_classifier.py implements the model used for the ESP Application

### Steps to recreate the working application: 
***

1. Clone the repository using git or you can directly download the zipped files.
2. Open the ![ESP Application][https://github.com/callmechristian/ESP/tree/master/ESP%20Application] folder.
3. Install the latest version of python and the required libraries like numpy, pandas, pickle, sklearn and flask.
4. Run the following command from your terminal to start the web application.
```console
python main.py
```
If all the required libraries were already installed, you could see the web application started running at the following URL
```console
http://127.0.0.1:5000/
```
(This application uses the port number of 5000 and runs in the localhost - 127.0.0.1 If you have any other application running at the port 50000, you can modify them in the main.py file)
