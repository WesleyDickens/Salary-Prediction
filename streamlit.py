import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pandas import Int64Index

jobs_cleaned = pd.read_csv('jobs_cleaned.csv')

med_P = np.nanpercentile(jobs_cleaned['med_salary'], [1, 99])
jobs_cleaned = jobs_cleaned[(jobs_cleaned['med_salary'] > med_P[0]) & (jobs_cleaned['med_salary'] < med_P[1])]

min_P = np.nanpercentile(jobs_cleaned['min_salary'], [1, 99])
jobs_cleaned = jobs_cleaned[(jobs_cleaned['min_salary'] > min_P[0]) & (jobs_cleaned['min_salary'] < min_P[1])]

max_P = np.nanpercentile(jobs_cleaned['max_salary'], [1, 99])
jobs_cleaned = jobs_cleaned[(jobs_cleaned['min_salary'] > max_P[0]) & (jobs_cleaned['min_salary'] < max_P[1])]

# Create a new LabelEncoder for each categorical column
title_encoder = LabelEncoder()
location_encoder = LabelEncoder()
experience_level_encoder = LabelEncoder()

# Fit and transform each column with its respective encoder
jobs_cleaned['title_enc'] = title_encoder.fit_transform(jobs_cleaned['title'])
jobs_cleaned['location_enc'] = location_encoder.fit_transform(jobs_cleaned['location'])
jobs_cleaned['formatted_experience_level_enc'] = experience_level_encoder.fit_transform(jobs_cleaned['formatted_experience_level'])


X = jobs_cleaned[['title_enc', 'location_enc','formatted_experience_level_enc']]

def train_salary_model(salary_level):
    # Select the salary level for prediction
    y = jobs_cleaned[salary_level]
    
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Initializing and training the XGBoost regressor model
    # After running a few grid searches I found that each model had the same best parameters
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.2,
                              max_depth = 5, n_estimators = 1000)
    xg_reg.fit(X_train, y_train)

    # Making predictions on the test set
    y_pred = xg_reg.predict(X_test)

    # Calculating the RMSE for the predictions
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    
    # Returning the trained model
    return xg_reg, salary_level, y_test, y_pred, rmse

def model_eval(model, salary_level, y_test, y_pred, rmse):

    st.write(f"Salary Level: {salary_level}")
    st.write(f"RMSE: {rmse}")

    # Plotting the scatter plot for Actual vs Predicted Salaries
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel('Actual Salaries')
    ax.set_ylabel('Predicted Salaries')
    ax.set_title(f'Actual vs. Predicted Salaries ({salary_level})')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

    # Displaying the plot in the Streamlit app
    st.pyplot(fig)
    
    
### JOB TITLE PREDICTION 

jobs = jobs_cleaned[['title','description']]
job_descriptions = jobs['description'].tolist()  # Use list for maintaining order
job_titles = jobs['title'].unique()  # This will not be used directly but ensures we're aligned with descriptions

# User input job description
# user_input_description = """
# I want to use SQL and Python to build data visualizations and solve problems with data analysis
# """

user_input_description = st.text_input("Enter job description (Leave blank to search all jobs): ")

if user_input_description:
    # TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform the job descriptions to TF-IDF matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(job_descriptions)

    # Transform the user input job description to TF-IDF vector
    user_input_vector = tfidf_vectorizer.transform([user_input_description])

    # Compute the cosine similarity between user input job description and each job description in the list
    cos_similarity = cosine_similarity(user_input_vector, tfidf_matrix)[0]

    # Find the indices of the top 5 cosine similarity scores
    top_5_indices = np.argsort(cos_similarity)[-5:]

    # Retrieve the job titles corresponding to these top 5 indices from the original DataFrame
    most_similar_job_titles = [jobs.iloc[i]['title'] for i in top_5_indices]

    # Reverse the list to have the most similar first
    most_similar_job_titles.reverse()

    title = st.selectbox("Choose a job title", most_similar_job_titles)
else:
    title = st.selectbox("Choose a job title", job_titles)



### SALARY PREDICTION 

med_salary_model, _, _, _, _ = train_salary_model('med_salary')
min_salary_model, _, _, _, _ = train_salary_model('min_salary')
max_salary_model, _, _, _, _ = train_salary_model('max_salary')

# Streamlit UI components for user input
st.title('Salary Prediction')

# Assuming jobs_cleaned['location'].unique() is an available list of locations
# title = st.selectbox('Select Title', options=jobs_cleaned['title'].unique())
experience_level = st.selectbox('Select Experience Level', options=jobs_cleaned['formatted_experience_level'].unique())
location = st.selectbox('Select Location', options=jobs_cleaned['location'].unique())

# For simplicity, assuming other inputs are fixed as per your example
user_input = {
    'title_enc': [f'{title}'],
    'location_enc': [f'{location}'],
    'formatted_experience_level_enc': [f'{experience_level}']
}

user_input_df = pd.DataFrame(user_input)

# Encoding the user inputs
user_input_df['title_enc'] = title_encoder.transform(user_input_df['title_enc'])
user_input_df['location_enc'] = location_encoder.transform(user_input_df['location_enc'])
user_input_df['formatted_experience_level_enc'] = experience_level_encoder.transform(user_input_df['formatted_experience_level_enc'])

# Predicting the salary for the user input
min_predicted_salary = min_salary_model.predict(user_input_df)
median_predicted_salary = med_salary_model.predict(user_input_df)
max_predicted_salary = max_salary_model.predict(user_input_df)

# Displaying the results
st.markdown(f"**----{location}-----**")
# st.write(f"\n Estimated salary: ${min_predicted_salary[0]:,.2f} - ${median_predicted_salary[0]:,.2f} - ${max_predicted_salary[0]:,.2f} \n")

# Displaying the median salary prominently using st.metric
st.metric(label="Median Estimated Salary", value=f"${median_predicted_salary[0]:,.2f}")

# Optionally, use st.markdown or st.write to display the full range more subtly
st.markdown(f"Estimated Salary Range: \${min_predicted_salary[0]:,.2f} - ${max_predicted_salary[0]:,.2f}")

### CITY PLOT

city_salaries_df = pd.DataFrame(columns=['city', 'min_predicted_salary', 'median_predicted_salary', 'max_predicted_salary'])

# Example user input
for location_ in jobs_cleaned['location'].unique():
    user_input = {
        'title_enc': [f'{title}'],
        'location_enc': [f'{location_}'],
        'formatted_experience_level_enc': [f'{experience_level}']
    }

    user_input_df = pd.DataFrame(user_input)

    user_input_df['title_enc'] = title_encoder.transform(user_input_df['title_enc'])
    user_input_df['location_enc'] = location_encoder.transform(user_input_df['location_enc'])
    user_input_df['formatted_experience_level_enc'] = experience_level_encoder.transform(user_input_df['formatted_experience_level_enc'])

    # Predict the salary for the user input
    city = location_
    min_predicted_salary = min_salary_model.predict(user_input_df)
    median_predicted_salary = med_salary_model.predict(user_input_df)
    max_predicted_salary = max_salary_model.predict(user_input_df)
    
    temp_df = pd.DataFrame({
        'city': [city],
        'min_predicted_salary': min_predicted_salary,
        'median_predicted_salary': median_predicted_salary,
        'max_predicted_salary': max_predicted_salary
    })
    
    city_salaries_df = pd.concat([city_salaries_df, temp_df], ignore_index=True)


top_cities = city_salaries_df.sort_values(by='median_predicted_salary', ascending=False).head(10)
top_cities = top_cities[top_cities.city != 'United States']

# Plotting
fig, ax = plt.subplots()
ax.bar(top_cities['city'], top_cities['median_predicted_salary'], color='skyblue')
ax.set_xlabel('city', fontsize=12)
ax.set_ylabel('Median Salary ($)', fontsize=12)
ax.set_title('Top 10 Median Salaries by City', fontsize=14)
ax.tick_params(axis='x', rotation=45)

# Display the plot in the Streamlit app
st.pyplot(fig)
    
