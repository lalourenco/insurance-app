import pickle
import os
import pandas as pd
import numpy as np
from flask import Flask,request,Response
from insurance.Insurance import Insurance

model=pickle.load(open('model/xgb_class.pkl','rb'))

app=Flask(__name__)
@app.route('/predict',methods=['POST'])

def health_insurance_predict():
    
    test_json=request.get_json()
    
    if test_json: 
        if isinstance(test_json,dict):
            test_raw=pd.DataFrame(test_json,index=[0])            
        else: 
            test_raw=pd.DataFrame(test_json,columns=test_json[0].keys())    
        pipeline=Insurance()
        df1=pipeline.data_cleaning(test_raw)
        df2=pipeline.feature_engineering(df1)
        df3=pipeline.data_preparation(df2)
        df_response=pipeline.get_prediction(model,test_raw,df3)
        return df_response
    
    else:
        return Response('{}',status=200,mimetype='application/json')
    
if __name__ == '__main__':
    port=os.environ.get('PORT',5000)
    app.run(host='0.0.0.0',port=port)