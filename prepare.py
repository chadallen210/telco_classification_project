##### IMPORTS #####
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

##### TELCO PREP #####
def telco_prep(df):
    
    '''
    This function take in the df and preps it for exploration.
    Actions taken in this function include: checking for duplicates, setting the index to customer_id, dealing with any spaces and null values, encoding columns, feature engineering, and renaming and dropping columns.
    '''
    
    # drop any duplicates
    df.drop_duplicates(inplace=True)
    
    # set index to customer_id
    df.set_index('customer_id', drop=True, inplace=True)
    
    # fill any empty spaces with np.nan
    df.replace(' ', np.nan, inplace=True)
    
    # drop rows that contain null values
    df.dropna(axis=0, inplace=True)
    
    # convert total_charges to a numeric data type
    df = df.astype({'total_charges': 'float64'})
    
    # rename 'tenure' to 'tenure_months'
    df = df.rename(columns={'tenure': 'tenure_months'})
    # add column for tenure in years
    df['tenure_years'] = df.tenure_months / 12
    
    # create dummy variable for gender - male=1, female=0
    gender_dummy = pd.get_dummies(df.gender, drop_first=True)
    # add dummy to telcodf
    df = pd.concat([df, gender_dummy], axis=1)
        
    # encode columns with 'Yes' and 'No'
    # new values (0 = No, 1 = Yes)
    df.partner = df.partner.replace({'No': 0, 'Yes': 1})
    df.dependents = df.dependents.replace({'No': 0, 'Yes': 1})
    df.phone_service = df.phone_service.replace({'No': 0, 'Yes': 1})
    df.paperless_billing = df.paperless_billing.replace({'No': 0, 'Yes': 1})
    df.churn = df.churn.replace({'No': 0, 'Yes': 1})
    
    # combine 'partner' + 'dependents' into 'family' 
    # new values(0 = none, 1 = partner OR dependent, 2 = both)
    df['family'] = df.partner + df.dependents
    
    # drop 'phone_service' and encode 'multiple_lines'
    df = df.drop(columns='phone_service')
    # encode values in 'multiple_lines'
    # new values (0 = no phone service, 1 = single line, 2 = multiple lines)
    df.multiple_lines = df.multiple_lines.replace({'No phone service': 0, 'No': 1, 'Yes': 2})
    # rename 'multiple_lines' to 'phone_service'
    df = df.rename(columns={'multiple_lines': 'phone_service'})
    
    # encode values in 'online_security' and 'online_backup'
    # new values (0 = No internet service, 0 = No, 1 = Yes)
    df.online_security = df.online_security.replace({'No internet service': 0, 'No': 0, 'Yes': 1})
    df.online_backup = df.online_backup.replace({'No internet service': 0, 'No': 0, 'Yes': 1})
    # combine 'online_security' and 'online_backup' into 'online_services'
    # new values(0 = no internet service, 1 = security OR backup, 2 = both)
    df['online_services'] = df.online_security + df.online_backup
        
    # encode values in 'device_protection' and 'tech_support'
    # new values (0 = No internet service, 0 = No, 1 = Yes)
    df.device_protection = df.device_protection.replace({'No internet service': 0, 'No': 0, 'Yes': 1})
    df.tech_support = df.tech_support.replace({'No internet service': 0, 'No': 0, 'Yes': 1})
    # combine 'device_protection' and 'tech_support' into 'support_services'
    # new values(0 = no internet service, 1 = device_protection OR tech_support, 2 = both)
    df['support_services'] = df.device_protection + df.tech_support
        
    # encode values in 'streaming_tv' and 'streaming_movies'
    # new values (0 = No internet service, 0 = No, 1 = Yes)
    df.streaming_tv = df.streaming_tv.replace({'No internet service': 0, 'No': 0, 'Yes': 1})
    df.streaming_movies = df.streaming_movies.replace({'No internet service': 0, 'No': 0, 'Yes': 1})
    # combine 'streaming_tv' and 'streaming_movies' into 'streaming_services'
    # new values(0 = no internet service, 1 = streaming_tv OR streaming_movies, 2 = both)
    df['streaming_services'] = df.streaming_tv + df.streaming_movies
        
    # convert 'payment_type_id' 
    # new values(0 = manual_pay, 1 = auto_pay)
    df.payment_type_id = df.payment_type_id.replace({1: 0, 2: 0, 3: 1, 4: 1})
    # rename 'payment_type_id' to 'auto_pay'
    df = df.rename(columns={'payment_type_id': 'auto_pay'})
    
    # convert 'contract_type'
    # new values(0 = 'Month-to-month', 1 = 'One year', 2 = 'Two year')
    df.contract_type = df.contract_type.replace({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    
    # convert 'internet_service_type'
    # new values(0 = 'None', 1 = 'DSL', 2 = 'Fiber optic')
    df.internet_service_type = df.internet_service_type.replace({'None': 0, 'DSL': 1, 'Fiber optic': 2})
    
    # drop columns
    df = df.drop(columns=['gender', 'partner', 'dependents', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'payment_type', 'contract_type_id', 'internet_service_type_id'])
    
    return df


##### TELCO SPLIT #####

def telco_split(df):
    '''
    This function takes in the telco data acquired by get_telco_data,
    performs a split and stratifies on churn column.
    (test = 20%, validate = 24%, train = 56% of the original dataset)
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                        random_state=1234, 
                                        stratify=df.churn)
    train, validate = train_test_split(train_validate, train_size=0.7, 
                                   random_state=1234, 
                                   stratify=train_validate.churn)
    return train, validate, test
