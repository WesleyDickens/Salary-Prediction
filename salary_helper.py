import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Load your cleaned jobs dataset
jobs_cleaned = pd.read_csv('/mount/src/salary-prediction/jobs_cleaned.csv',low_memory=False)

# Initialize LabelEncoders
title_encoder = LabelEncoder()
location_encoder = LabelEncoder()
skill_name_encoder = LabelEncoder()
experience_level_encoder = LabelEncoder()

# Fit LabelEncoders
jobs_cleaned['title_enc'] = title_encoder.fit_transform(jobs_cleaned['title'])
jobs_cleaned['location_enc'] = location_encoder.fit_transform(jobs_cleaned['location'])
jobs_cleaned['skill_name_enc'] = skill_name_encoder.fit_transform(jobs_cleaned['skill_name'])
jobs_cleaned['formatted_experience_level_enc'] = experience_level_encoder.fit_transform(jobs_cleaned['formatted_experience_level'])

# Load models
xgb = xgb.XGBRegressor()

loaded_med_model = xgb.load_model('med_model.json')
loaded_min_model = xgb.load_model('min_model.json')
loaded_max_model = xgb.load_model('max_model.json')

# Streamlit app starts here
st.title('Salary Prediction for Job Roles')

# User inputs
role = st.text_input('What job title do you want?')
location = 'New York, NY'  # Assuming static location for simplicity
skill_name = st.selectbox('Select your skill area:', jobs_cleaned['skill_name'].unique())
experience_level = st.selectbox('Select your experience level:', jobs_cleaned['formatted_experience_level'].unique())

if st.button('Predict Salary'):
    user_input = pd.DataFrame({
        'title': [role],
        'location': [location],
        'skill_name': [skill_name],
        'formatted_experience_level': [experience_level]
    })

    # Transform user input using the trained LabelEncoders
    user_input['title_enc'] = title_encoder.transform(user_input['title'])
    user_input['location_enc'] = location_encoder.transform(user_input['location'])
    user_input['skill_name_enc'] = skill_name_encoder.transform(user_input['skill_name'])
    user_input['formatted_experience_level_enc'] = experience_level_encoder.transform(user_input['formatted_experience_level'])

    # Drop non-encoded columns
    user_input.drop(columns=['title', 'location', 'skill_name', 'formatted_experience_level'], inplace=True)

    # Predict the salary for the user input
    min_predicted_salary = loaded_min_model.predict(user_input)
    median_predicted_salary = loaded_med_model.predict(user_input)
    max_predicted_salary = loaded_max_model.predict(user_input)

    # Display predicted salary
    st.write(f"----{location}-----")
    st.write(f"\n Estimated salary: ${min_predicted_salary[0]:,.2f} - ${median_predicted_salary[0]:,.2f} - ${max_predicted_salary[0]:,.2f} \n")

# Note: Ensure all models and csv files are in the same directory as the streamlit app or adjust paths accordingly.
