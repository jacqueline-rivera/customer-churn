# Objective
Customer attrition, or customer churn, is the percentage of customers that stop using a product within a given time frame. The first goal of this project is to identify important features that help determine if a customer will churn. The second goal of this project is to build a model that will predict if a customer will churn. 

# Data
The Telco Customer Churn dataset is utilized in this project and can be found [here](https://www.kaggle.com/blastchar/telco-customer-churn). This dataset contains 7,043 unique records with 21 features:
* Customer demographic features:
  * _customerID_
  * _gender_ - 'Male' or 'Female'
  * _SeniorCitizen_ - 1 if customer is 65 or older, 0 if not
  * _Partner_ - 'Yes' if customer has a partner, 'No' if not
  * _Dependents_ - 'Yes' if customer lives with dependents, 'No' if not
  
* Service options features:
  * _PhoneService_ - 'Yes' if a customer signed up for home phone service, 'No' if not
  * _MultipleLines_ - 'Yes' if a customer subscribed to multiple telephone lines, 'No' if not, 'No phone service' if not applicable
  * _InternetService_ - 'DSL', 'Fiber optic', 'No' if customer did not subscribe to internet service
  * _OnlineSecurity_ - 'Yes' if a customer subscribed for this option, 'No' if not, 'No internet service' if not applicable
  * _OnlineBackup_ - 'Yes' if a customer subscribed for this option, 'No' if not, 'No internet service' if not applicable
  * _DeviceProtection_ - 'Yes' if a customer subscribed for this option, 'No' if not, 'No internet service' if not applicable
  * _TechSupport_ - 'Yes' if a customer subscribed for this option, 'No' if not, 'No internet service' if not applicable
  * _StreamingTV_ - 'Yes' if a customer uses the internet service to stream TV, 'No' if not, 'No internet service' if not applicable
  * _StreamingMovies_ - 'Yes' if a customer uses the internet service to stream movies, 'No' if not, 'No internet service' if not applicable

* Customer account features:
  * _tenure_ - integer, the number of months the customer has been with the company
  * _Contract_ - 'Month-to-month', 'One year', 'Two year'
  * _PaperlessBilling_ - 'Yes' if the customer enrolled in paperless billing, 'No' if not
  * _PaymentMethod_ - 'Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'
  * _MonthlyCharges_ - float, current total monthly charge for all services
  * _TotalCharges_ - float, current total charges calculated at the end of the quarter

* Target Variable:
  * _Churn_ - 'Yes' if customer left the company this quarter, 'No' if not

# Data Preparation
Checking the data types tells us that the _TotalCharges_ feature is of the object data type instead of float64. The code below revealed that there are 11 blanks in _TotalCharges_. These blanks were converted to NaN and the rows that contained null values were dropped. The resulting data frame contains 7,032 records.

```python
# check what is causing the object data type
print([x for x in data['TotalCharges'] if any(char.isdigit() for char in x) == False])
```

# EDA
There is an imbalance in the target variable:
* Customers that did not churn: 5163 or approximately 73%
* Customers that did churn: 1869 or approximately 23%

<p align="center">
  <img src="https://user-images.githubusercontent.com/71897317/128102632-1640f4cd-ea20-4d1c-8e97-328d22a6baa6.png"/>
</p>

<!--![churndistribution](https://user-images.githubusercontent.com/71897317/128102632-1640f4cd-ea20-4d1c-8e97-328d22a6baa6.png)-->

The imbalance will be addressed after taking a look at the other features in the dataset. Below we have the relationship between churn rate and _tenure_. The churn rate is calculated by dividing the number of churns by the total number of customers for each unique value of _tenure_. We see that there is a negative correlation between churn rate and _tenure_. This suggests that the longer a customer has been with the company, the less likely the customer will churn.

<p align="center">
  <img src="https://user-images.githubusercontent.com/71897317/128102861-5776b9f4-c6bc-403d-b144-9e2c551e2815.png"/>
</p>

<!--![churnvtenure](https://user-images.githubusercontent.com/71897317/128102861-5776b9f4-c6bc-403d-b144-9e2c551e2815.png)--> 

<p align="center">
  <img src="https://user-images.githubusercontent.com/71897317/128102941-f33dcb0f-313a-4a43-8c6f-dab645ffc900.png"/>
</p>

<!--![tenure-monthlycharges-totalcharges](https://user-images.githubusercontent.com/71897317/128102941-f33dcb0f-313a-4a43-8c6f-dab645ffc900.png)-->

Above we have the pairwise relationships between _Churn_ and the numerical features. There is a difference in variance, medians, 25th percentile and 75th percentile within each plot. It appears that these features may be relevant when investigating churn. We can use two-sample t-tests to test whether the means for each group within the features are different. The p-values for all three t-tests were nearly 0 therefore there is enough evidence to conclude there is a difference in the means. 

```python
# t-tests
from pingouin import ttest

no = data[data['Churn']=='No']
yes = data[data['Churn']=='Yes']

print('Tenure:', ttest(no['tenure'], yes['tenure'])['p-val'])
print('')
print('Monthly Charges:', ttest(no['MonthlyCharges'], yes['MonthlyCharges'])['p-val'])
print('')
print('Total Charges:', ttest(no['TotalCharges'], yes['TotalCharges'])['p-val'])
```

Churn rate for demographic features: 

gender | No | Yes | Churn %
-------|----|-----|--------
Female | 2544 | 939 | 26.96
Male | 2619 | 930 | 26.20

SeniorCitizen | No | Yes | Churn %
--------------|----|-----|--------
False | 4497 | 1393 | 23.65
True | 666 | 476 | 41.68

Partner | No | Yes | Churn %
--------|----|-----|--------
False | 2439 | 1200 | 32.98
True | 2724 | 669 | 19.72

Dependents | No | Yes | Churn %
-----------|----|-----|--------
False | 3390 | 1543 | 31.28
True | 1773 | 326 | 15.53

The churn rate for the rest of the variables are below. We can see that...

![part1](https://user-images.githubusercontent.com/71897317/128103680-0c45c84b-fd4a-494a-8a59-9290584f5bac.png)

![part2](https://user-images.githubusercontent.com/71897317/128103792-fb9bfd61-53df-46d1-8292-3757746396e0.png)


# Feature Selection with scikit-learn

# Model Selection with PyCaret
