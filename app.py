from flask import Flask, render_template,request
import pandas as pd
import pickle
import api
import numpy as np
from selected_model import knn

app = Flask(__name__)

# Importing the pickle files
model = pickle.load(open('/Users/kranthireddy/Documents/OU 3rd SEM/Data Mining/Final/Project/knn_regressor.pkl','rb'))
list1 = pickle.load(open('/Users/kranthireddy/Documents/OU 3rd SEM/Data Mining/Final/Project/features.pkl','rb'))
r2 = pickle.load(open('/Users/kranthireddy/Documents/OU 3rd SEM/Data Mining/Final/Project/r2.pkl','rb'))
confi_knn = pickle.load(open('/Users/kranthireddy/Documents/OU 3rd SEM/Data Mining/Final/Project/CI_1.pkl','rb'))

# Function to display the Landing page
@app.route('/')
def main():
    return render_template('home.html')

# Function to display Yield recommender page
@app.route('/yielder')
def yieldform():
    return render_template('yield_recommender.html')

# Function to display about section
@app.route('/about')
def aboutform():
    return render_template('about.html')

# API endpoint function to get the country and Crop type data
@app.route('/apiendpoint')
def getinfo():
    return api.apiEnd()

@app.route('/predict', methods=['GET','POST'])
def predict_data(): 
    # Getting the user entered data   
    country= request.form.get('selectcountry')
    Rainfall = request.form.get('rainfall')
    temp= request.form.get('temperature')
    pesti= request.form.get('pesticide')
    crop= request.form.get('selectcrop')
    a=[country,crop,pesti,Rainfall,temp]
    data_clean=list1
    # Copying the user entered data into original data
    data_clean.loc[len(data_clean)]=a
    print(data_clean)
    # Converting complete data into numerical format
    data_clean=pd.get_dummies(data_clean, columns=['Area',"Item"], prefix = ['Country',"Item"])
    # Selecting the required features for predicting using loc method
    r=data_clean.loc[len(data_clean)-1,:].values
    test=np.array(r).reshape(-1, len(r))
    test=test.astype(float)
    # Predicting using the knn model
    ee=model.predict(test)
    print(ee)
    return render_template('predict.html', prediction =int(ee[0]), R2=round(float(r2),3), confidence=confi_knn)

app.run(debug=True,port=8000)