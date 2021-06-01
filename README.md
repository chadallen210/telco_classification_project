## Telco Churn Classification Project

###### Chad Allen
###### 01 June 2021

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Project Summary
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

#### Project Objectives
> - Document code, process (data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways in a Jupyter Notebook report.
> - Create modules (acquire.py, prepare.py) that make your process repeateable.
> - Construct a model to predict customer churn using classification techniques.
> - Deliver a 5 minute presentation consisting of a high-level notebook walkthrough using your Jupyter Notebook from above; your presentation should be appropriate for your target audience.
> - Answer panel questions about your code, process, findings and key takeaways, and model.

#### Business Goals
> - Find drivers for customer churn at Telco.
> - Construct a ML classification model that accurately predicts customer churn.
> - Document your process well enough to be presented or read like a report.

#### Audience
> - Codeup Data Science team

#### Project Deliverables
> - A Jupyter Final Report Notebook 
> - A README.md file
> - All necessary modules to make project reproducible
> - A csv file of customer_id, probability of churn, and predictions.

#### Project Context
> - The telco dataset came from the Codeup database.


#### Data Dictionary

The telco database contains four tables. The tables were joined together into a single pandas DataFrame for this project.

After preparing the data, the remaining features and values are listed below:

| Feature               | Data Type | Description                                                | Values                                                                   |
|-----------------------|-----------|------------------------------------------------------------|--------------------------------------------------------------------------|
| customer_id           | object    | Unique identifier assigned to each customer                | Set as index                                                             |
| auto_pay              | int64     | Indicates if a customer uses a form of auto_pay            | 0 = manual_pay, 1 = auto_pay                                             |
| senior_citizen        | int64     | Indicates if a customer is a senior citizen                | 0 = No, 1 = Yes                                                          |
| tenure_months         | int64     | # of months as a customer                                  |                                                                          |
| phone_service         | int64     | Indicates the type of phone service the customer has       | 0 = no phone service, 1 = single line, 2 = multiple lines                |
| paperless_billing     | int64     | Indicates if a customer uses paperless billing             | 0 = No, 1 = Yes                                                          |
| monthly_charges       | float64   | Monthly charges for customer                               |                                                                          |
| total_charges         | float64   | Total charges for customer                                 |                                                                          |
| churn                 | int64     | Indicates if customer has ended service                    | 0 = No, 1 = Yes                                                          |
| contract_type         | int64     | Indicates the type of contract the customer has            | 0 = no phone service, 1 = single line, 2 = multiple lines                |
| internet_service_type | int64     | Indicates the type of internet service the customer has    | 0 = 'None', 1 = 'DSL', 2 = 'Fiber optic'                                 |
| tenure_years          | float64   | # of years as a customer                                   |                                                                          |
| Male                  | uint8     | Indicates if the customer is "male"                        | 0 = No, 1 = Yes                                                          |
| family                | int64     | Combined feature of "partner" and "dependents"             | 0 = none, 1 = partner OR dependent, 2 = both                             |
| online_services       | int64     | Combined feature of "online_security" and "online_backup"  | 0 = no internet service, 1 = security OR backup, 2 = both                |
| support_services      | int64     | Combined feature of "device_protection" and "tech_support" | 0 = no internet service, 1 = device_protection OR tech_support, 2 = both |
| streaming_services    | int64     | Combined feature of "streaming_tv" and "streaming_movies"  | 0 = no internet service, 1 = streaming_tv OR streaming_movies, 2 = both  |

#### Initial Hypotheses

> - **Hypothesis 1 -** Rejected the Null Hypothesis; the type of internet service appears to affect churn.
> - alpha = .05
> - $H_0$: The type of internet service has no affect on churn.  
> - $H_a$: The type of internet service does have an affect on churn.

> - **Hypothesis 2 -** Rejected the Null Hypothesis; having auto-pay appears to have an affect on churn.
> - alpha = .05
> - $H_0$: Having a form of auto-pay has no affect on churn. 
> - $H_a$: Having a form of auto-pay has no affect on churn. 

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Executive Summary - Conclusions & Next Steps
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

> - Created 3 classification models - LogisticRegression, DecisionTree, and RandomForest - and tested them with a list of 6 features - 'auto_pay', 'tenure_months', 'paperless_billing', 'contract_type', 'internet_service_type', 'family' - each having at least 20% correlation.
> - Each model predicted customer churn equally well on the train dataset, between 79-80%, using the listed features.
> - Chose the LogisticRegression model as the best model with a 80% accuracy rate for predicting my target value, churn. This model outperformed my baseline score of 73% accuracy, so it has value.
> - Initial exploration and statistical testing revealed that the selected features produced well-fit models, and with more time exploring additional features and/or adjusting hyperparameters could improve the results.

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Pipeline Stages Breakdown

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

##### Project Planning
- create a README file with project and business goals, a data dictionary, ideas and hypotheses, and how to recreate the project
- acquire the data, create a function(s) to automate the process, create acquire.py file containing the function(s), import and utilize the automated processes in the final notebook
- clean and prepare the data, create a function(s) to automate the process, create a prepare.py file containing the function(s), import and utilize the automated processes in the final notebook
- create visuals of variable distributions, create hypotheses, run statistical tests, reject or fail to reject null hypotheses, and document findings and takeaways
- establish baseline predictions, create, fit and evaluate models on the train and validate datasets, document findings and takeaways
- choose the best model and evaluate on the test dataset, document conclusions, takeaways, and recommendations
- create a predictions.csv with customer_id, probability of churn, and predictions

___

##### Data Acquisition
> - Store functions that are needed to acquire data from telco database on the Codeup data science database server; make sure the acquire.py module contains the necessary imports to run my code.
> - The function will return a pandas DataFrame.
> - Import the get_telco_data function from the acquire.py module and use it to acquire the data in the Final Report Notebook.
> - Complete some initial data summarization (`.info()`, `.describe()`, `.value_counts()`, ...).
___

##### Data Preparation
> - Store functions needed to prepare the telco data; make sure the module contains the necessary imports to run the code. The final function should do the following:
    - Handle any missing values.
    - Handle erroneous data and/or outliers that need addressing.
    - Encode variables as needed.
    - Create any new features, if made for this project.
    - Split the data into train/validate/test.
> - Import the telco_prep function from the prepare.py module and use it to prepare the data in the Final Report Notebook.
> - Import the telco_split function from the prepare.py module and use it to split the data into train, validate and test subsets in the Final Report Notebook.
___

##### Data Exploration and Analysis

> - Using the prepared data, discover correlations, determine which features to examine further.
> - Create visualizations and run basic tests to help narrow down a list of variables to run statisical tests on.
> - Form and test hypotheses, then run at least 2 statistical tests. Document hypotheses, set an alpha, run the tests, and document takeaways.
> - Summarize my conclusions, provide clear answers to my specific questions, and summarize any takeaways/action plan from the work above.
___

##### Modeling and Evaluation
> - Establish a baseline accuracy to determine if having a model is better than no model and train and compare at least 3 different models. Document these steps well.
> - Train (fit, transform, evaluate) multiple models, varying the algorithm and/or hyperparameters you use.
> - Compare evaluation metrics across all the models you train and select the ones you want to evaluate using your validate dataframe.
> - Feature Selection (after initial iteration through pipeline): Are there any variables that seem to provide limited to no additional information? If so, remove them.
> - Based on the evaluation of the models using the train and validate datasets, choose the best model to try with the test data, once.
> - Test the final model on the out-of-sample data (the testing dataset), summarize the performance, interpret and document the results.
___

##### Delivery
> - Introduce myself and my project goals at the very beginning of my notebook walkthrough.
> - Summarize my findings at the beginning like I would for an Executive Summary. (Don't throw everything out that I learned from Storytelling) .
> - Walk Codeup Data Science Team through the analysis I did to answer my questions and that lead to my findings. (Visualize relationships and Document takeaways.) 
> - Clearly call out the questions and answers I am analyzing as well as offer insights and recommendations based on my findings.

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Reproduce My Project

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

You will need your own env file with database credentials along with all the necessary files listed below to run my final project notebook. 
- [x] Read this README.md
- [ ] Download the aquire.py, prepare.py, explore.py and telco_classification_final_report.ipynb files into your working directory
- [ ] Add your own env file to your directory. (username, password, host)
- [ ] Run the telco_classification_final_report.ipynb notebook