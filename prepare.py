##### IMPORTS #####
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

##### TELCO PREP #####




##### TELCO SPLIT #####

def telco_split(df):
    '''
    This function takes in the telco data acquired by get_telco_data,
    performs a split and stratifies on churn column.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                        random_state=1234, 
                                        stratify=df.churn)
    train, validate = train_test_split(train_validate, train_size=0.7, 
                                   random_state=1234, 
                                   stratify=train_validate.churn)
    return train, validate, test
