# Objective
Customer attrition, or customer churn, is the percentage of customers that stop using a product within a given time frame. The first goal of this project is to identify important features that help determine if a customer will churn. The second goal of this project is to build a model that will predict if a customer will churn. 

# Data
The Telco Customer Churn dataset is utilized in this project and can be found [here](https://www.kaggle.com/blastchar/telco-customer-churn). This dataset contains 7,043 unique records with the following features:
* Customer demographic features:
  * customerID
  * gender - 'Male' or 'Female'
  * SeniorCitizen - 1 if customer is a senior citizen, 0 if not
  * Partner - 'Yes' if customer has a partner, 'No' if not
  * Dependents - 'Yes' if customer has dependents, 'No' if not
  
* Service options features:
  * PhoneService - 'Yes' if a customer signed up for this option, 'No' if not
  * MultipleLines - 'Yes' if a customer signed up for this option, 'No' if not, 'No phone service' if not applicable
  * InternetService - 'DSL', 'Fiber optic', 'No' if not applicable
  * OnlineSecurity - 'Yes' if a customer signed up for this option, 'No' if not, 'No internet service' if not applicable
 * OnlineBackup - 'Yes' if a customer signed up for this option, 'No' if not, 'No internet service' if not applicable
 * DeviceProtection - 'Yes' if a customer signed up for this option, 'No' if not, 'No internet service' if not applicable
 * TechSupport - 'Yes' if a customer signed up for this option, 'No' if not, 'No internet service' if not applicable
 * StreamingTV - 'Yes' if a customer signed up for this option, 'No' if not, 'No internet service' if not applicable
 * StreamingMovies - 'Yes' if a customer signed up for this option, 'No' if not, 'No internet service' if not applicable


* tenure (integer, the number of months
* Contract
* PaperlessBilling
* PaymentMethod
* MonthlyCharges
* TotalCharges
* Churn - Customer left within the last month (Yes/No)

# Data Preparation

# EDA

# Feature Selection with scikit-learn

# Model Selection with PyCaret
