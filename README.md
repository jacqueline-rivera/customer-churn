# Objective
Customer attrition, or customer churn, is the percentage of customers that stop using a product within a given time frame. The first goal of this project is to identify important features that help determine if a customer will churn. The second goal of this project is to build a model that will predict if a customer will churn. 

# Data
The Telco Customer Churn dataset is utilized in this project and can be found [here](https://www.kaggle.com/blastchar/telco-customer-churn). This dataset contains 7,043 unique records with the following features:
* Customer demographic features:
  * _customerID_
  * _gender_ - 'Male' or 'Female'
  * _SeniorCitizen_ - 1 if customer is 65 or older, 0 if not
  * _Partner_ - 'Yes' if customer has a partner, 'No' if not
  * _Dependents_ - 'Yes' if customer lives with dependents, 'No' if not
  
* Service options features:
  * _PhoneService_ - 'Yes' if a customer signed up for home phone service, 'No' if not
  * MultipleLines - 'Yes' if a customer subscribed to multiple telephone lines, 'No' if not, 'No phone service' if not applicable
  * InternetService - 'DSL', 'Fiber optic', 'No' if customer did not subscribe to internet service
  * OnlineSecurity - 'Yes' if a customer subscribed for this option, 'No' if not, 'No internet service' if not applicable
  * OnlineBackup - 'Yes' if a customer subscribed for this option, 'No' if not, 'No internet service' if not applicable
  * DeviceProtection - 'Yes' if a customer subscribed for this option, 'No' if not, 'No internet service' if not applicable
  * TechSupport - 'Yes' if a customer subscribed for this option, 'No' if not, 'No internet service' if not applicable
  * StreamingTV - 'Yes' if a customer uses the internet service to stream TV, 'No' if not, 'No internet service' if not applicable
  * StreamingMovies - 'Yes' if a customer uses the internet service to stream movies, 'No' if not, 'No internet service' if not applicable

* Customer account features:
  * tenure - integer, the number of months the customer has been with the company
  * Contract - 'Month-to-month', 'One year', 'Two year'
  * PaperlessBilling - 'Yes' if the customer enrolled in paperless billing, 'No' if not
  * PaymentMethod - 'Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'
  * MonthlyCharges - float, current total monthly charge for all services
  * TotalCharges - float, current total charges calculated at the end of the quarter

* Target Variable:
  * Churn - 'Yes' if customer left the company this quarter, 'No' if not

# Data Preparation

# EDA

# Feature Selection with scikit-learn

# Model Selection with PyCaret
