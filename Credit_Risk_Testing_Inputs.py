import warnings
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the pipeline (which includes preprocessing and model)
pipeline = joblib.load('datasets/best_credit_score_model.pickle')
cleaned_unscaled_data = pd.read_csv("cleaned_unscaled_data.csv")

warnings.filterwarnings("ignore")

print("Hi! Welcome, if you are a bank, credit card institution, or an organization that lends,\nyou can test if a user from your organization is likely to default with a high accuracy of 95%!\n")


# person_age = input("Enter the person's age: ")
# person_income = input("Enter the person's income: ")
# person_home_ownership = input("Enter the person's home ownership type: ")
# person_emp_length = input("Enter the person's employee length in years: ")
# loan_intent = input("Enter the person's loan intent: ")
# loan_grade = input("Enter the person's loan grade: ")
# loan_amnt = input("Enter the person's amount of loan taken: ")
# loan_int_rate = input("Enter the person's interest rate on the loan taken: ")
# loan_percent_income = input("Enter the person's income in percent on the loan taken: ")
# cb_person_default_on_file = input("Enter if person's has deafulted in the past with Y ot N: ")
# cb_person_cred_hist_length = input("Enter the person's credit history length: ")


# Input data
person_age = 20
person_income = 23000
person_home_ownership = "OWN"
person_emp_length = 10.0
loan_intent = "MEDICAL"
loan_grade = "E"
loan_amnt = 3500
loan_int_rate = 1.14
loan_percent_income = 0.5
cb_person_default_on_file = "Y"
cb_person_cred_hist_length = 10

# Define category-to-label mappings
home_ownership_mapping = {'MORTGAGE': 0, 'OTHER': 1, 'OWN': 2, 'RENT': 3}
default_on_file_mapping = {'N': 0, 'Y': 1}
loan_grade_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
loan_intent_mapping = {'DEBTCONSOLIDATION': 0, 'EDUCATION': 1, 'HOMEIMPROVEMENT': 2, 'MEDICAL': 3, 'PERSONAL': 4, 'VENTURE': 5}

# Encode categorical columns using the predefined mappings
person_home_ownership_encoded = home_ownership_mapping.get(person_home_ownership, -1)
cb_person_default_on_file_encoded = default_on_file_mapping.get(cb_person_default_on_file, -1)
loan_grade_encoded = loan_grade_mapping.get(loan_grade, -1)
loan_intent_encoded = loan_intent_mapping.get(loan_intent, -1)

# Correct order of columns as per the trained model
column_names = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
                'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                'person_home_ownership_encoded', 'cb_person_default_on_file_encoded',
                'loan_grade_encoded', 'loan_intent_encoded']


# Create DataFrame for input data with encoded columns in the correct order
input_data = pd.DataFrame([[person_age, person_income, person_emp_length, loan_amnt,
                            loan_int_rate, loan_percent_income, cb_person_cred_hist_length,
                            person_home_ownership_encoded, cb_person_default_on_file_encoded,
                            loan_grade_encoded, loan_intent_encoded]],
                          columns=column_names)

new_data = cleaned_unscaled_data.append(input_data, ignore_index=True)

numeric_columns = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
                                   'loan_percent_income', 'cb_person_cred_hist_length']
# this function scales all numeric data except categorical data

def column_scaler(dataframe, column_names):
    scaler = StandardScaler()

# Apply the scaler only to the numeric columns
    dataframe[column_names] = scaler.fit_transform(dataframe[column_names])

    return dataframe


column_scaler(new_data, numeric_columns)


most_recent_row = new_data.tail(1)

most_recent_row = most_recent_row.drop(columns=['loan_status'])

# Print or return the last row
#print(most_recent_row)

# Make prediction using the pipeline, which should include preprocessing
prediction = pipeline.predict(most_recent_row)


# Print the result
if prediction == 1:
    print("The applicant is likely to default.")
else:
    print("The applicant is not likely to default.")
