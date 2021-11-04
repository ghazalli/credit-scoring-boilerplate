from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, StandardScaler, LabelEncoder
from sklearn_pandas import DataFrameMapper
import time

#Initialize the flask App
app = Flask(__name__)

# LOAD MODEL
model = joblib.load('model/best_model.pkl', 'r')
mapper_fit = joblib.load('model/mapper_fit.pkl', 'r')

numerical_cols=['sub_grade_num', 'short_emp', 'emp_length_num','dti', 'payment_inc_ratio', 'delinq_2yrs', \
                'delinq_2yrs_zero', 'inq_last_6mths', 'last_delinq_none', 'last_major_derog_none', 'open_acc',\
                'pub_rec', 'pub_rec_zero','revol_util']

categorical_cols=['grade', 'home_ownership', 'purpose']

col = numerical_cols + categorical_cols

# # BQ TABLE
# projectid = 'baf-demo-323109'

# table_schema = [{'name':'name', 'type':'STRING'},
#                 {'name':'grade', 'type':'STRING'},
#                 {'name':'sub_grade_num', 'type':'FLOAT64'},
#                 {'name':'short_emp', 'type':'INTEGER'},
#                 {'name':'emp_length_num', 'type':'INTEGER'},
#                 {'name':'home_ownership', 'type':'STRING'}, 
#                 {'name':'dti', 'type':'FLOAT64'}, 
#                 {'name':'purpose', 'type':'STRING'}, 
#                 {'name':'payment_inc_ratio', 'type':'FLOAT64'}, 
#                 {'name':'delinq_2yrs', 'type':'FLOAT64'}, 
#                 {'name':'delinq_2yrs_zero', 'type':'FLOAT64'}, 
#                 {'name':'inq_last_6mths', 'type':'FLOAT64'}, 
#                 {'name':'last_delinq_none', 'type':'INTEGER'}, 
#                 {'name':'last_major_derog_none', 'type':'INTEGER'}, 
#                 {'name':'open_acc', 'type':'FLOAT64'}, 
#                 {'name':'pub_rec', 'type':'FLOAT64'}, 
#                 {'name':'pub_rec_zero', 'type':'INTEGER'},  
#                 {'name':'revol_util', 'type':'FLOAT64'},  
#                 {'name':'prediction', 'type':'STRING'},  
#                 {'name':'prediction_good_loan', 'type':'FLOAT64'},  
#                 {'name':'prediction_bad_loan', 'type':'FLOAT64'},  
#                 {'name':'input_time', 'type':'TIME'},  
#                 {'name':'input_date', 'type':'DATE'}]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    data = [x for x in request.form.values()]
    colz = [x for x in request.form.keys()]
    dfx = pd.DataFrame(data=[data], columns=colz)
    
    XX1 = mapper_fit.transform(dfx[col])
    XX2 = dfx[numerical_cols]
    XX = np.hstack((XX1,XX2))
    
    # PREDICT MODEL
    prediction = model.predict(XX)
    prediction_good_loan = model.predict_proba(XX)[:,0][0]
    prediction_bad_loan = model.predict_proba(XX)[:,1][0]

    input_date = str(time.strftime('%Y-%m-%d'))
    input_time = str(time.strftime('%H:%M:%S'))

    dfx["prediction"] = int(prediction)
    di = {1: "Bad Loan", 0: "Good Loan"}
    dfx["prediction"] = dfx["prediction"].map(di)
    
    dfx["prediction_good_loan"] = round(prediction_good_loan*100,1)
    dfx["prediction_bad_loan"] = round(prediction_bad_loan*100,1)
    
    dfx["input_time"] = input_time
    dfx["input_date"] = input_date
    
    # new_input_path = 'new_input/new_input.csv'
    
    # if os.path.exists(new_input_path) == True:
    #     old_data = pd.read_csv(new_input_path,index_col=False)
    # else:
    #     old_data = pd.DataFrame()
        
    # updated_data = old_data.append(dfx,ignore_index = True)
    # updated_data.to_csv(new_input_path,header=True,index=False)

    # dfx.to_gbq(destination_table = 'tabular_case.demo_credit_scoring_new_input',
    #     project_id = projectid,
    #     if_exists='append',
    #     table_schema=table_schema)
    
    if round(prediction_good_loan*100,1) < 60:
      suggestion = "reject"
    elif (round(prediction_good_loan*100,1) >= 60 and round(prediction_good_loan*100,1) < 70):
      suggestion = "accept with further analysis"
    elif (round(prediction_good_loan*100,1) >= 70 and round(prediction_good_loan*100,1) < 85):
      suggestion = "approve"
    else:
      suggestion = "approve"
  
    return render_template("predict.html",
                           prediction_text_suggest = "We {strong} suggest you to {suggestion} this application. Credit risk would be predicted as {loan} with probability {prob}%".format(
                           suggestion = suggestion,
                           strong = "strongly" if round(prediction_good_loan*100,1) > 85 else "",
                           loan = "Bad Loan (default)" if int(prediction) == 1 else "Good Loan",
                           prob = round(prediction_bad_loan*100,1) if int(prediction) == 1 else round(prediction_good_loan*100,1)))

if __name__ == '__main__':
    app.run()