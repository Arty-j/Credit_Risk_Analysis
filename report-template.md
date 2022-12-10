# Module 12 Credit Risk Analysis Report

## Overview of the Analysis

The purpose of this data analysis was to use machine learning to build a model that helps determine creditworthiness of future loan applicants. 

Historical data from a lending service is used to train a model on patterns between the health of a loan, and all the other data they collect on their loan applicants. The health of a loan indicates creditworthiness of the borrower. Machine Learning algorithms use this process of finding numerical patterns and relationships in the data to 'fit' a model which can intake new data in the future and predict the target outcome with significant accuracy.  In this case, lending services, are looking to identify a 'target' class of loan applicants are not-credit worthy and are likely to default on a loan. 

## Analysis Process
A logistical regression model is used here to predict whether customers fall into one of two classes (binary classification):
    - ‘0’ - healthy loan = creditworthy
    - ‘1’ - defaulted loan = not-creditworthy

1- The lender's historical customer-loan data is loaded into a dataframe and examined for shape, and layout.  The columns are reassigned names to fit the parameters of the model. The “loan status” column is reassigned to our target variable ‘y’, and all the other columns are assigned to the features variable ‘X’.
    -`y.value_counts()` function is used to see clearly the exact number of values for ‘0’ vs ‘1’ in our ‘y’ variable data. In this dataset, the ‘0’ creditworthy class far outsizes the ‘1’ not-credit worthy class. This sort of imbalance in data points may lead to model bias, as it is training itself to identify more ‘0’. 

2- The data is separated into a training set (about 80%), and a testing set (about 20%), using the sklearn library.  The library calculates 4 variables, which contain our separated data for fitting the model (the ‘X’ & ‘y’ training sets), and testing the model’s accuracy (‘X’ & ‘y’ testing sets). 

3- Using the Logistic Regression model in the sklearn library the model is algorithmically fit to the training data. The model uses numeric relationships to determine which data points are catagorized as ‘0’ and which are categorized as ‘1’.  It is then used to predict an outcome of ‘0’ or ‘1’ on our testing dataset.

4- Running a classification report (from the sklearn library), compares the model’s predictions on the testing set to the actual ‘0’ vs ‘1’ values.  

5- Because of the imbalanced size in the dataset classes, the training data was resampled using Random Over Sampler module from the imbalanced-learn library.  This resampled our smaller class repeat times, so that the final size of the two classes was equal, hopefully reducing any bias in the model.
    -`y_resampled.value_counts()`, was again used to verify the ‘0’ and ‘1’ data points in the resampled set were of equal size

6-The Logistic Regression model was refit with the resampled training data, then a new set of predictions was run using the original testing set and compared to the actual results in the testing set.

## Results

* Logistic Regression Model with Original data:
    * This model had an Accuracy Score of 94% showing it’s overall ability to predict the outcome of an applicant being creditworthy and paying off their loan, vs being not-creditworthy and defaulting on their loan, was pretty good. 
    
    * The Precision score, or the percent of the predictions that were correct, for this model was :
        - near 100% (.9966) on its predictions of class ‘0’ (creditworthy=healthy loan)
        - 85% on it’s predictions of class ‘1’ (not-creditworthy = unhealthy loan), meaning it misclassified about 15% of applicants who were creditworthy, as not-creditworthy. 

    * The Recall, or percentage of positive cases (target cases) that the model caught, came out at .90, so the model correctly predicted 90% of the “not-creditworthy=default likely” loan applicants (‘1’ class), but 10% of the cases where the applicant is likely to default, they were classisied as creditworthy.

    * The overall F1 score, or the mean of the precision and the recall, of this model was near 100%, for it’s ability to predict creditworthiness, but it only correctly predicted a non-creditworthy applicant about 87% of the time, which is not ideal for risk mitigation.


* Logistic Regression Model with Resampled data:
**To see if we could improve the accuracy, precision and Recall results of our model we adjusted the size of class ‘1’ so that it would have an equal number of data points as class ‘0’. Hopefully reducing any bias in the model.**
    * This model had an Accuracy Score of 99% showing it’s ability to predict the outcome of an applicant being creditworthy and paying off their loan, vs being not-creditworthy and defaulting on their loan, was excellent.

    * The Precision score, or the percent of the predictions that were correct, for this model was :
        - near 100% (.9966) on its predictions of class ‘0’ (creditworthy=healthy loan)
        - 84% of it's overall predictions of class ‘1’ (not-creditworthy = unhealhty loan) were correct, meaning it misclassified about 16% of applicants who were creditworthy, as not-creditworthy. (while not ideal, not a lending risk)

    * The Recall, or percentage of positive cases (target cases) that the model caught, came out at .99, so the model correctly predicted 99% of the “not-creditworthy=default likely” loan applicants (‘1’ class). 

    *The overall F1 score, or the mean of the precision and the recall, of this model was near 100%, for it’s ability to predict creditworthiness, but it only correctly predicted a non-creditworthy applicant about 91% of the time.

## Summary
The purpose of creating this model is for risk mitigation on behalf of the lender: to predict the loan applicants who are not creditworthy and likely to default on their loan. There are far fewer instances of loan default vs loan payoff happening in real life, and thus the dataset as it comes from the lending institution is inherently imbalanced.  Using this imbalanced dataset to train our model, makes it excellent for predicting the larger class of creditworthy applicants, but it is not well suited to identify the few, ‘target’ cases, of those customers who are not creditworthy and likely to default on a loan, increasing the lending instituions risk of loss.

By resampling the data of our smaller ‘target’ class to equal that of the larger class (by using a resampling algorithm), the Logistic Regression model was trained on a balanced number of datapoints, significantly improving its prediction accuracy of our target class. While precision on predictions of unhealthy loans (non-credit worthy applicants) dipped slightly in this run, incorrectly catagorizing an additional 1% of applicants as not-creditworthy when they did in-fact have healthy loans, this does not pose a financial risk to the institution, it may simply decrease their potential revenue. 

This resampled data along with the logistic regression model run was much better at correctly predicting the more important ‘target’ class of unhealthy loans, thus identifying 99% of applicants who are not-credit worthyand would likely default on their loans.  Correct identification of these customers is the best risk mitigation strategy for the lending instition, and a best-fit model for them.

