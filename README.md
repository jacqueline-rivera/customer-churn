Customer attrition, or customer churn, is the percentage of customers that stop using a product within a given time frame. The goals of this project are to identify important features that help determine if a customer will churn and to build a model that will predict if a customer will churn. 

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

# EDA
## Data Types
Checking the data types tells us that the _TotalCharges_ feature is of the object data type instead of float64. The code below revealed that there are 11 blanks in _TotalCharges_; these blanks are converted to NaN. 

```python
# check what is causing the object data type
print([x for x in df['TotalCharges'] if any(char.isdigit() for char in x) == False])
```
```python
# replace blanks with NaN
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = df['TotalCharges'].astype('float64')
```

## Target Variable
There is an imbalance in the target variable that will be addressed after taking a look at the features in the dataset:
* Customers that did not churn: 5174 or approximately 73%
* Customers that did churn: 1869 or approximately 27%

<p align="center">
  <img src="https://user-images.githubusercontent.com/71897317/129407180-597f927c-373b-42b6-bd72-92fba58b6a7c.png"/>
</p>
<!--![churndistribution](https://user-images.githubusercontent.com/71897317/129407180-597f927c-373b-42b6-bd72-92fba58b6a7c.png)-->

## Numerical Features
We will examine the numerical features first. Below we have the relationship between churn rate and _tenure_. The churn rate is calculated by dividing the number of churns by the total number of customers for each unique value of _tenure_. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/71897317/128102861-5776b9f4-c6bc-403d-b144-9e2c551e2815.png"/>
</p>
<!--![churnvtenure](https://user-images.githubusercontent.com/71897317/128102861-5776b9f4-c6bc-403d-b144-9e2c551e2815.png)--> 

We see that there is a negative correlation between churn rate and _tenure_. This suggests that the longer a customer has been with the company, the less likely the customer will churn. For _MonthlyCharges_ and _TotalCharges_, we can compare the distributions using layered histograms.

<p align="center">
  <img src="https://user-images.githubusercontent.com/71897317/129407964-d845048a-480e-42dd-b58f-96ec369f0b07.png"/>
</p>
<!--![layeredhist](https://user-images.githubusercontent.com/71897317/129407964-d845048a-480e-42dd-b58f-96ec369f0b07.png)-->

We can see in _MonthlyCharges_ that there is a sharp peak for customers that do not churn at the $20 range, otherwise the distributions for _Churn_ = 'Yes' and _Churn_ = 'No' follow a similar pattern for this feature. The distributions for _TotalCharges_ are both skewed right. 

Now that we have analyzed the numerical features, we can fill in missing values for _TotalCharges_:
```python
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
```

## Categorical Features
### Demographics

Next we can take a look at the point plots and churn rate breakdown for demographic features: 

![pointplots](https://user-images.githubusercontent.com/71897317/129410000-5ee6b716-4e0e-4a31-ac77-a6827777c8c8.png)
<!--![democharts](https://user-images.githubusercontent.com/71897317/128217083-35d88026-0f7e-4572-b16a-74288fd7ed0a.png)-->

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

* _gender_: appears to be the only demographic feature where the churn rate for each class are not so different 
* _SeniorCitizen_: customers who are 65 or older are approximately 2.3 times more likely to churn than customers who are not 
* _Partner_: customers who do not have a partner are approximately 2 times more likely to churn than customers who do have partners 
* _Dependents_: customers that do not have dependents are 2.4 times more likely to churn than customers who live with dependents 

### Service Options
Below we have the churn rates for the values in the service options features. We can see that the churn rate varies for the categories within the features. For instance, the 'Fiber optic' option in _InternetService_ has a churn rate that is at least 10% higher than the other values for this feature. 

![service](https://user-images.githubusercontent.com/71897317/129410555-bc1f50e9-ab94-477e-aaba-c3cfabfbddf2.png)

We can also look at the relationship between various features by creating a stratified contingency table. Here are a couple of examples:

<p align="center">
  <img src="https://user-images.githubusercontent.com/71897317/129411733-f5614aef-2865-4451-9ec6-631dad26e132.png"/>
</p>
<!--![techsupport-internetservice](https://user-images.githubusercontent.com/71897317/129411733-f5614aef-2865-4451-9ec6-631dad26e132.png)-->

First we have the stratified contingency table of _TechSupport_ and _InternetService_. Customers that have 'No Tech Support' and the 'Fiber optic' option of _InternetService_ have a churn rate of 49.37%. That is pretty high and it may be worth the company looking into. On the other hand, customers that have 'Tech Support' and the 'DSL' option of _InternetService_ have a churn rate of 9.86%. Why is there a low churn rate for this combination of features? Can something be learned from this and applied to customers that have 'Fiber optic' with 'No Tech Support'?

<p align="center">
  <img src="https://user-images.githubusercontent.com/71897317/129411919-03b8b950-51a4-41c7-bc06-37f7137eb12f.png"/>
</p>
<!--![security-backup](https://user-images.githubusercontent.com/71897317/129411919-03b8b950-51a4-41c7-bc06-37f7137eb12f.png)-->

Next we have the stratified contingency table of _OnlineSecurity_ and _OnlineBackup_. Here we see a stark difference between customers who have 'No Online Security' and 'No Online Backup' and customers who have 'Online Security' and 'Online Backup'. Is there a way to encourage 'Online Security' and 'Online Backup' in order to lower the churn rate?

### Customer Account 
Repeating the steps made for the service options features, below we have the churn rates for the values in the customer account features. 

![account](https://user-images.githubusercontent.com/71897317/129412654-6c261b20-829e-4a31-8ca2-460488461249.png)

Again, we can see that the curn rate varies for the categories within the features. For instance, the 'Electronic check' option in _PaymentMethod_ has a churn rate that is at least 25% higher than the other values in _PaymentMethod_ and the 'Month-to-month' option in _Contract_ has a churn rate that is at least 30% higher that the other values in _Contract_. Let's look at a couple of stratified contingency tables for this set of variables: 

<p align="center">
  <img src="https://user-images.githubusercontent.com/71897317/129412433-dd3f7327-5d75-408f-b854-f73d83ce5699.png"/>
</p>
<!--![billing-payment](https://user-images.githubusercontent.com/71897317/129412433-dd3f7327-5d75-408f-b854-f73d83ce5699.png)-->

First we have the table for _PaperlessBilling_ and _PaymentMethod_. We can see that there is an overall pattern of paperless billing leading to higher churn rates. Below we have the table for _Contract_ and _PaymentMethod_. 'Month-to-month' contracts paid with 'Electronic check' have a high churn rate of almost 54% while the same contract paith with a 'Mailed check' has a significantly lower churn rate of 31.58%.

<p align="center">
  <img src="https://user-images.githubusercontent.com/71897317/129412499-6d7eaa2f-5a74-44db-80e1-9de685f8361f.png"/>
</p>
<!--![contract-payment](https://user-images.githubusercontent.com/71897317/129412499-6d7eaa2f-5a74-44db-80e1-9de685f8361f.png)-->


## Dealing with Imbalanced Target Varible

Now we can address the imbalanced target variable. In this dataset there are 5,174 customers that did not churn and 1,869 customers that did churn. We will take two steps to try to overcome the imbalance: 
1. Use f1-score to measure the accuracy of the models
2. Combine random oversampling and random undersampling 

```python
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

X_df = df.drop('Churn', axis=1).copy()
y_df = df['Churn'].values

ros = RandomOverSampler(sampling_strategy=0.6)
X_ros, y_ros = ros.fit_resample(X_df, y_df)

rus = RandomUnderSampler(sampling_strategy=0.8)
X_co, y_co = rus.fit_resample(X_ros, y_ros)

sampling_df = X_co.copy()
sampling_df['Churn'] = y_co
```

This resulted in a new dataset that consists of 6,984 records with 3,880 customers that did not churn and 3,104 customers that did churn. Information on the f1-score can be found [here](https://deepai.org/machine-learning-glossary-and-terms/f-score). A tutorial for random oversampling and undersampling can be found [here](https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/).

# Data Cleaning

Now that we have an overview of the variables in the dataset and have modified the dataset to overcome the imbalanced target feature, we can encode our data. First we split the dataset into X (independent variables) and y (target variable), then we can encode all of the categorical features. We have seven categorical features that are binary and will be encoded using label encoding: _Churn (our y), gender, SeniorCitizen, Partner, Dependents, PhoneService_, and _PaperlessBilling_. The remaining categorical features will be encoded using one-hot-encoding: _MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract_, and _PaymentMethod_. We then split the dataset into training and testing sets using the train_test_split function from scikit-learn and scale the features as well:

```python
from sklearn import preprocessing

X_df = sampling_df.iloc[:, 1:-1].copy()
y = sampling_df.iloc[:, -1].values

# encoding binary categorical variables:
binary_col = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
              'PhoneService', 'PaperlessBilling']

for col in binary_col:
    le = preprocessing.LabelEncoder()
    X_df[col] = le.fit_transform(X_df[col])

# one-hot-encoding for the remaining categorical variables:
remaining_col = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

X_df = pd.get_dummies(X_df, columns=remaining_col, drop_first=True)
X = X_df.values

# class encoding:
le_class = preprocessing.LabelEncoder()
y = le_class.fit_transform(y)
```

```python
from sklearn.model_selection import train_test_split

# 60/40 train/test split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# 20/20 test/validation split:
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

# scale features
stdsc = preprocessing.StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
X_val_std = stdsc.transform(X_val)

# we want to preserve the column names for later
X_train_std = pd.DataFrame(X_train_std)
X_train_std.columns = X_df.columns
```

# Model Building

<!--
Next we can reduce the dimensionality of the dataset and select features that will result in the most optimal model using sequential backward selection (SBS). Information on SBS can be found [here](https://vitalflux.com/sequential-backward-feature-selection-python-example/). We run SBS on each model and plot the f1-score that was calculated as SBS removed features.

### Logistic Regression
The first model to be trained is Logistic Regression:

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=1)
lr.fit(X_train_std, y_train)
lr_pred = lr.predict(X_test_std)
```

The f1-score with all features for this model is 0.739. The classification report and confusion matrix are as follows:

 | | precision | recall | f1-score | support
 ---------|-----------|--------|----------|---------
 0 | 0.80 | 0.77 | 0.78 | 1162
 1 | 0.72 | 0.75 | 0.74 | 929
 accuracy | | | 0.76 | 2091
 macro avg | 0.76 | 0.76 | 0.76 | 2091
 weighted avg | 0.76 | 0.76 | 0.76 | 2091
 
 <p align="center">
  <img src="https://user-images.githubusercontent.com/71897317/128237256-736f38d0-6240-4829-afc2-b634eaa41ccb.png" width="350"/>
</p>
 <!--![lr-cm](https://user-images.githubusercontent.com/71897317/128237256-736f38d0-6240-4829-afc2-b634eaa41ccb.png)-->

<!--

We see that that the true label 1, or 'Yes', was incorrectly predicted as 0, or 'No', 228 times. We also see that the true label 0 was incorrectly predicted as 1 268 times. Running SBS on the trained Logistic Regression model and plotting the f1-score will allow us to choose the optimal number of features for our model. Here is the resulting plot from SBS on Logistic Regression: 

<p align="center">
  <img src="https://user-images.githubusercontent.com/71897317/128228179-b87171c8-358b-425b-9176-223781554440.png"/>
</p>
<!--![lr-SBS](https://user-images.githubusercontent.com/71897317/128228179-b87171c8-358b-425b-9176-223781554440.png)-->

<!--

It appears that 9 features is the lowest number of features that will result in the highest f1-score. We can take a look at what those 9 features are by printing the indices from SBS and finding the corresponding features from the data frame. The features are _SeniorCitizen, tenure, PhoneService, TotalCharges, OnlineBackup_Online Backup, TechSupport_Tech Support, Contract_One year, Contract_Two year_, and _PaymentMethod_Electronic check_.

### Support Vector Machine

We can repeat the process for the Support Vector Machine (SVM) model: 

```python
from sklearn.svm import SVC

svm = SVC(random_state=1)
svm.fit(X_train_std, y_train)
svm_pred = svm.predict(X_test_std)
```
The f1-score with all features for this model is 0.741. The classification report and confusion matrix are as follows:

 | | precision | recall | f1-score | support
 ---------|-----------|--------|----------|---------
 0 | 0.80 | 0.77 | 0.78 | 1162
 1 | 0.72 | 0.76 | 0.74 | 929
 accuracy | | | 0.76 | 2091
 macro avg | 0.76 | 0.76 | 0.76 | 2091
 weighted avg | 0.77 | 0.76 | 0.76 | 2091
 
<p align="center">
  <img src="https://user-images.githubusercontent.com/71897317/128239777-255f73e0-0564-409f-b6a5-2a2431850409.png" width="350"/>
</p>
 <!--![svm-cm](https://user-images.githubusercontent.com/71897317/128239777-255f73e0-0564-409f-b6a5-2a2431850409.png)-->

<!--

There is not much difference in the metrics between the SVM model and Logistic Regression model. From the confusion matrix, we see that that the true label 1, or 'Yes', was incorrectly predicted as 0, or 'No', 222 times. We also see that the true label 0 was incorrectly predicted as 1 273 times. Running SBS on the trained SVM model and plotting the f1-score will allow us to choose the optimal number of features for our model. Here is the resulting plot for SBS on SVM: 
 
<p align="center">
  <img src="https://user-images.githubusercontent.com/71897317/128228250-c5c029a9-bc2e-4c73-90f5-c881b3c58c8e.png"/>
</p>
<!--![svm-SBS](https://user-images.githubusercontent.com/71897317/128228250-c5c029a9-bc2e-4c73-90f5-c881b3c58c8e.png)-->

<!--

It appears that 12 features is the lowest number of features that will result in the highest f1-score. The features are _SeniorCitizen, Partner, tenure, PaperlessBilling, MonthlyCharges, MultipleLines_Single Line, DeviceProtection_No Device Protection, TechSupport_Tech Support, Contract_One year, Contract_Two year, PaymentMethod_Credit card (automatic)_, and _PaymentMethod_Electronic check_.

### Random Forest Classifier

We repeat the steps one final time for Random Forest Classifier model:

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=1)
rf.fit(X_train_std, y_train)
rf_pred = rf.predict(X_test_std)
```
The f1-score with all features for this model is 0.804. The classification report and confusion matrix are as follows:

 | | precision | recall | f1-score | support
 ---------|-----------|--------|----------|---------
 0 | 0.86 | 0.81 | 0.83 | 1162
 1 | 0.78 | 0.83 | 0.80 | 929
 accuracy | | | 0.82 | 2091
 macro avg | 0.82 | 0.82 | 0.82 | 2091
 weighted avg | 0.82 | 0.82 | 0.82 | 2091

<p align="center">
  <img src="https://user-images.githubusercontent.com/71897317/128240798-b442d31f-9d1f-4577-95b9-fe5b62c5ffcf.png" width="350"/>
</p>
<!--![rf-cm](https://user-images.githubusercontent.com/71897317/128240798-b442d31f-9d1f-4577-95b9-fe5b62c5ffcf.png)-->

<!--

With the Random Forest Classifier model, there is an improvement in the f1-score as well as accuracy. From the confusion matrix, we see that that the true label 1, or 'Yes', was incorrectly predicted as 0, or 'No', 158 times. We also see that the true label 0 was incorrectly predicted as 1 218 times. Running SBS on the trained Random Forest Classifier model and plotting the f1-score will allow us to choose the optimal number of features for our model. Here is the resulting plot for SBS on Random Forest: 

<p align="center">
  <img src="https://user-images.githubusercontent.com/71897317/128228363-22fd559c-03e2-45fd-b793-d71f31ddb4cd.png"/>
</p>
<!--![rf-SBS](https://user-images.githubusercontent.com/71897317/128228363-22fd559c-03e2-45fd-b793-d71f31ddb4cd.png)-->

<!--

It appears that 16 features is the lowest number of features that will result in the highest f1-score. The features are _SeniorCitizen, Partner, PaperlessBilling, MonthlyCharges, TotalCharges, MultipleLines_Single Line, InternetService_No internet service, OnlineBackup_Online Back up, DeviceProtection_No internet service, StreamingTV_Not Streaming TV, StreamingTV_Streaming TV, StreamingMovies_Not Streaming Movies, StreamingMoves_Streaming Movies, Contract_Two year, PaymentMethod_Credit card (automatic)_, and _PaymentMethod_Electronic check_.

**Features that were chose across all classifiers: _Contract_Two year, PaymentMethod_Electronic check_, and _SeniorCitizen_.**

**Features that did not appear in any of the chosen optimal models: _Dependents, InternetService_Fiber optic, MultipleLines_No phone service, OnlineBackup_No internet service, OnlineSecurity_No internet service, OnlineSecurity_Online Security, PaymentMethod_Mailed check, TechSupport_No internet service_, and, _gender_.**

*It is worth noting that the features that were chosen/not chosen across all classifiers may change depending on the records selected in the oversampling/undersampling step.* 

# Model Selection with PyCaret
For the second goal of this project, PyCaret is employed to train various models, choose the best model, tune the chosen model, and see how it performs. Implementing PyCaret:

```python
from pycaret.classification import *
clf = setup(data, target='Churn', ignore_features=['customerID']
best_model = compare_models(sort='F1')
```
PyCaret classification is used on both the original data and the dataset created through oversampling and undersampling in order to compare the results. Below, we have the output for _compare_models_ on the original dataset on the left and the output for _compare_models_ on the sampling dataset on the right. We can see that the highest f1-score for the original datset was 0.6250 with the Naive Bayes model and the highest f1-score for the sampling dataset was 0.7991 with the Random Forest Classifier.

<p align="center">
  <img src="https://user-images.githubusercontent.com/71897317/128246747-1db36807-8af8-454d-940e-907900716520.png" width="300"/>
  <img src="https://user-images.githubusercontent.com/71897317/128245796-fcde595b-0143-4deb-9c97-045c4cc45a44.png" width="300"/>
</p>
<!--![compare](https://user-images.githubusercontent.com/71897317/128246747-1db36807-8af8-454d-940e-907900716520.png)
![compare_samp](https://user-images.githubusercontent.com/71897317/128245796-fcde595b-0143-4deb-9c97-045c4cc45a44.png)-->

<!--

Looking at the confusion matrices, we can see how the sampling dataset resulted in better classifications for the dataset. Below, we have the confusion matrix for the original dataset on the left and the confusion matrix for the sampling dataset on the right.

<p align="center">
  <img src="https://user-images.githubusercontent.com/71897317/128246973-f2670e67-d45d-48b8-9361-a864016a6c77.png" width="300"/>
  <img src="https://user-images.githubusercontent.com/71897317/128246070-22f7c82f-3920-4f8e-a955-ea2a4ba89459.png" width="300"/>
</p>
<!--![confmat](https://user-images.githubusercontent.com/71897317/128246973-f2670e67-d45d-48b8-9361-a864016a6c77.png)
![confmat_samp](https://user-images.githubusercontent.com/71897317/128246070-22f7c82f-3920-4f8e-a955-ea2a4ba89459.png)-->

<!--

Although the f1-score was the metric focused on throughout the project, overall accuracy is worth taking a look at using ROC curves. Below we have the ROC curve for the original dataset on the left and the ROC curve for the sampling dataset on the right. We can see that the dataset created with oversampling and undersampling had better accuracy. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/71897317/128247063-2bd32b6b-235f-4988-af06-b3d668f18959.png" width="300"/>
  <img src="https://user-images.githubusercontent.com/71897317/128246171-01161e49-e222-4f4c-8e68-ed6e0096efdd.png" width="300"/>
</p>
<!--![AUC](https://user-images.githubusercontent.com/71897317/128247063-2bd32b6b-235f-4988-af06-b3d668f18959.png)
![AUC_samp](https://user-images.githubusercontent.com/71897317/128246171-01161e49-e222-4f4c-8e68-ed6e0096efdd.png)-->

**Overall, Random Forest Classifier with data that was oversampled and undersampled performed the best with an f1-score of 80%.**
