import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# cleaned_data = pd.read_csv("cleaned_data.csv")
# numeric_columns = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
#                                    'loan_percent_income', 'cb_person_cred_hist_length']

def column_scaler(dataframe, column_names):
    scaler = StandardScaler()

# Apply the scaler only to the numeric columns
    dataframe[column_names] = scaler.fit_transform(dataframe[column_names])

    return dataframe


