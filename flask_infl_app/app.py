from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np
app = Flask(__name__)
#Load the pickle model
model = pickle.load(open("pklmodel.pkl","rb"))

"""Home Page"""
@app.route("/")
def home():
    return render_template('index.html')

"""Predict function"""
@app.route("/predict",methods=["GET","POST"])
def predict():
    regionalmember = request.form.get('regionalmember')
    year = request.form.get('year')
    subregion = request.form.get('subregion')
    countrycode = request.form.get('countrycode')
    
    #Predicting single values from a model
    single_test  = [[regionalmember,year,subregion,countrycode]]
    single_test = pd.DataFrame(single_test)
    #Increase shape
    desired_shape = (1,104)
# Check if the desired shape is larger than the initial shape
    if desired_shape[1] > single_test.shape[1]:
        # Calculate the number of additional columns needed
        num_additional_columns = desired_shape[1] - single_test.shape[1]
    
        # Create a new array with the desired shape, filled with zeros
        expanded_array = np.pad(single_test, ((0, 0), (0, num_additional_columns)), mode='constant')
        #One hot encoding to a dataframe

        expanded_array_df = pd.DataFrame(expanded_array)
        encoded_single = pd.get_dummies(expanded_array_df)
        encoded_single = encoded_single.values
        predice = model.predict(encoded_single)
        predice = str(predice).replace('[', '').replace(']', '')
        print("predice",type(predice))
    return render_template('index.html',display_in_page_pred=predice)
    

if __name__ == '__main__':
    app.run(debug=True)