import pickle
import inflection
import pandas as pd

class Insurance:
    
    def __init__(self):
        self.age_scaler=pickle.load(open('parameter/age_scaler.pkl','rb'))
        self.annual_premium_scaler=pickle.load(open('parameter/annual_premium_scaler.pkl','rb'))
        self.vintage_scaler=pickle.load(open('parameter/vintage_scaler.pkl','rb'))
        self.region_code_encoding=pickle.load(open('parameter/region_code_encoding.pkl','rb'))
        self.policy_sales_channel_encoding=pickle.load(open('parameter/policy_sales_channel_encoding.pkl','rb'))
        
    def data_cleaning(self,data):
        columns_name=['id','Gender','Age','Driving_License','Region_Code','Previously_Insured',
                      'Vehicle_Age','Vehicle_Damage','Annual_Premium','Policy_Sales_Channel',
                      'Vintage','Response']
        data.columns=list(map(lambda x: inflection.underscore(x),columns_name))
        data['gender']=list(map(lambda x: inflection.underscore(x),data['gender']))
        return data
    
    def feature_engineering(self,data):
        data['vehicle_damage']=data['vehicle_damage'].apply(lambda x: 0 if x=='No' else 1)
        data['vehicle_age']=data['vehicle_age'].apply(lambda x: 'below_1_year' if x=='< 1 Year' else 'between_1_2_year' if x=='1-2 Year' else 'over_2_year')
        data['gender']=data['gender'].apply(lambda x: 0 if x=='male' else 1)
        return data
    
    def data_preparation(self,data):
        data['age']=self.age_scaler.fit_transform(data[['age']].values)
        data['annual_premium']=self.annual_premium_scaler.fit_transform(data[['annual_premium']].values)
        data['vintage']=self.vintage_scaler.fit_transform(data[['vintage']].values)
        data.loc[:,'region_code']=data['region_code'].map(self.region_code_encoding)
        data=pd.get_dummies(data,prefix=['vehicle_age'],columns=['vehicle_age'])
        data.loc[:,'policy_sales_channel']=data['policy_sales_channel'].map(self.policy_sales_channel_encoding)
        features_selected=['vintage','annual_premium','age','region_code','policy_sales_channel','vehicle_damage','previously_insured']
        return data[features_selected]
    
    def get_prediction(self,model,original_data,test_data):
        pred=model.predict_proba(test_data)
        original_data['prediction']=pred[:,1].tolist()
        return original_data.to_json(orient='records',date_format='iso')    