
# Phase 3 Project

Pump it Up: Data Mining the Water Table Competition
Hosted by DrivenData

## Introduction

An exploratory data analysis of Tanzanian water well data collected by Taarifa and the Tanzanian Ministry of Water and provided by [Driven Data](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/) 

My goal in this analysis is to extract insight from patterns in the dataset features and model them using classification techniques to predict whether a given well is functional, non functional, or needing repair. Being able to predict the condition of water wells and when they may fail is an important facet of preventative maintainance in order to provide clean water to as many Tanzanian communities as possible. 

## Technologies
* There are several Jupyter Notebooks found in this repo that contain the data analysis and modeling used in this project and they were created using Python 3.7.6
* There is a keynote presentation for non-technical audiences available under the file name "Phase3Presentation.pdf"

### Necessary libraries to reproduce and run this project are:

* Pandas 
* NumPy
* MatPlotLib
* Seaborn
* SciPy
* DateTime
* SKLearn
* StatsModels
* XGBoost
* CatBoost

## Objectives

* Explore and analyze dataset using visualizations to identify outliers and feature distributions among the target variable
* Clean data by removing or imputing missing or null values and changing any necessary data types 
* Analyze independent variables for the strongest possible variance among distributions of the target variable
* Engineer features in order to be processed and modeled by classification techniques using feature scaling and one-hot encoding
* Model dependent/independent variable relationship iteratively using different classification techniques
* Validate model performance and ability to predict well functionality 

## Methodology

There are many classification techniques available to data scientists and how succesful their outcome is depends on choosing the most suitable technique given the dataset you're modeling and the problem you're trying to solve. Real life classification problems can use Accuracy, Precision, Recall, and F1 Score as evaluation metrics and the choice of the metric chosen depends on the problem you're solving for. Each of those metrics has a trade off in scoring based on ratios of false positives, true positives, false negatives, and true negatives. 

Accuracy is the most fitting metric for this dataset due to the multi-class outcome, meaning a well can be functional, non-functional, or functional-needs repair. Precision and recall can only evaluate two classes at once so they are not viable for multi-class datasets. Accuracy in this instance will be a measure of how many wells are correctly predicted to be functional, non-functional, and needing repair to the overall number of wells in the dataset. If we were to take the class of wells that are functional-needs repair out of the data, precision would be a measure of how many wells are predicted to be functional out of the total number of correctly predicted functional and incorrectly predicted functional wells. Optimizing to precision would then reduce the amount of wells that are predicted to be functional but are actually non-functional (false positive), but could increase the amount of wells that are predicted to be non-functional and are actually functional(false negative). Recall would be a measure of how many wells are predicted to be functional to the total number of correctly predicted functional and actually functional but predicted non-functional (false negative). Optimizing to recall could reduce the amount of wells that are predicted non-functional but actually functional(false negatives), but would likely increase the number of wells that predicted to be functional and are actually non-functional (false positives). F1 score seeks to balance the the number of false positives and false negatives, but can only evaluate two outcomes at a time as well. 

Some well known classification modeling techniques are Logistic Regression, K-Nearest-Neighbors, Support Vector Machines, Decision Trees, Bagged (Bootstrap Aggregated) Decision Trees, Random Forests, Adaptive Boosted Trees, Gradient Boosted Trees, and Extreme Gradient Boosted Trees. Due to the dataset's dimensions of 59,000 rows and 300+ columns that are primarily comprised of categorical features I will be focusing my time on a few tree based learners because they are known to handle high dimension datasets well, and tuning their hyperparameters as opposed to experimenting with all model types.

## Table of Contents

* [Exploratory Data Analysis](#EDA)
* [Feature Engineering](#Features)
* [Modeling](#Models)
* [Conclusions](#Conclusion)

<a name="EDA"></a>
### Exploratory Data Analysis

The dataset provided by DrivenData is comprised of 59,400 rows of examples of waterpoints and 40 original feature columns. The feature set included multiple columns with null/missing values and a few data types that needed to be changed. All of the columns with missing values were categorical so I imputed them into an "other" category instead of dropping them to keep as much predictive power as possible in the dataset. 
Several categorical features had over 2,000 unique values. Leaving that many values in the data would have made the feature space too dimensional to process efficiently and would have led to over fitting the models so those feature's values were reduced by manually grouping entries with syntax errors together and then retaining a number of the most common entries and placing the least represented into an 'other' category. 
I used bar plots to compare the distribution of outcomes among feature values to ensure there was enough variance in outcome to justify retaing the feature. Two features were dropped due to a uniform distribution, meaning they had an equal amount of functional and non-functional wells, which doesn't provide the model with any useful information. 
Finally, after plotting each feature individually I found that several features contained redundant information that just had different value name orders or were grouped differently, so the redundant features with less variance among outcomes were dropped.

<a name="Features"></a>
### Feature Engineering

Once the dataset was was cleaned it was ready for preprocessing. Due to the high number of categorical columns present, and the amount of values to be one hot encoded in each I didn't attempt to engineer any new features. After the categorical columns were one-hot encoded (binarizing each feature value as it's own column) the feature space increased to 347 columns. 
Many machine learning models perform better with scaled data so I used MinMaxScaler to scale an instance of the training data and compare the results.  


<a name="Models"></a>
### Modeling

Working on a data science project is an iterative process and that is especially true for the modeling phase. Model performance is very sensitive to hyper-parameter settings and model parameters are sensitive to the input data. My approach for this project was to instantiate several different naive models (all hyper-parameters set to defualt) and then fine tune the hyper-parameters of the naive model using search methods and manual comparisons to achieve the best performance. 

The following are the titles and scores of the naive models.

Decision Tree: 75.47% Accuracy
![decisiontree](https://raw.githubusercontent.com/joshblumer/dsc-phase-3-project/main/images/decisiontree.png)

Bagged Tree: 79.30% Accuracy
![baggedtree](https://raw.githubusercontent.com/joshblumer/dsc-phase-3-project/main/images/baggedtree.png)

Random Forest: 67.39% Accuracy
![randomforest](https://raw.githubusercontent.com/joshblumer/dsc-phase-3-project/main/images/randomforest.png)

XGBoost: 74.71% Accuracy
![xgboost](https://raw.githubusercontent.com/joshblumer/dsc-phase-3-project/main/images/xgboost.png)

The Bagged Tree model had the highest test score but was highly over-fit and did not generalize well to the test data and the XGBoost model was not near as over-fit so next I instantiated a GridSearch Cross Validation search to check ranges of hyper-parameter values for the best model fit. 

The first GridSearch returned an estimator that scored 80.21% Accuracy
![gridsearch1](https://raw.githubusercontent.com/joshblumer/dsc-phase-3-project/main/images/gridsearch1.png)

After the improvement from the first GridSearch I expanded the range of values to check in a second GridSearch.

The second GridSearch returned an estimator that scored 81.06% Accuracy. 
![gridsearch2](https://raw.githubusercontent.com/joshblumer/dsc-phase-3-project/main/images/gridsearch2.png)

After tuning the hyper-parameters of the XGBoost model I went back to the training data to see if there were any fluctuations that could be made to improve the models performance. I experimented with different value counts in the categorical feature columns that contained many values, dropped continuous features with very heavily skewed distributions, and tried a model with none of the columns removed that I had originally removed, but none improved on the initial data preprocessing model scores. 

I was able to improve the accuracy of the model by MinMax Scaling the data first to 81.17% Accuracy. 
![minmax](https://raw.githubusercontent.com/joshblumer/dsc-phase-3-project/main/images/minmax.png)

I also attempted to use SMOTE(Synthetic Minority Over-Sampling Technique) due to the class imbalance of the target variable. It improved the training score but reduced the test score to 78.59% Accuracy.
![smote](https://raw.githubusercontent.com/joshblumer/dsc-phase-3-project/main/images/smote.png)

The last step in the modeling portion of the project was to process the test file, model it using the best performing model I had achieved, and submit it to DrivenData for scoring. The test data file varied from the training file so I experimented with different column value counts with it as well, but the best score returned came from the same model that performed best on the training file which was the second GridSearchCV hyper-parameter settings combined with MinMaxScaling the data. 

The highest score achieved in the models submitted to DrivenData was 80.93% Accuracy.
![drivendata](https://raw.githubusercontent.com/joshblumer/dsc-phase-3-project/main/images/high_score.png)

<a name="Conclusion"></a>
### Conclusions

I chose this dataset for this project due to it being an active competition and wanting to gain the valuable experience of participating in one while still a data science student. 
From the scope of approaching this dataset as a classification task that my only goal was to achieve as high of an accuracy score as possible, I've learned a great deal that will make me a more proficient and skilled data scientist. The first and most important lesson I learned throughout the process of this project was to utilize functions and pipelines. Iterating through data features and models manually takes up much more time and space than what is required. Functions are more difficult for inexperienced data scientists to code but they are exponentially more efficient than processing data manually. 
This is the first time I've worked with data that had the training and testing values broken up into separate files. In working with this format I learned the valuable lesson of doing EDA on all of your data at once together. I initially cleaned and preprocessed the training file and then modeled that, but when I went back to model the test file the model performed differently because the test data was not the same as the training data as I assumed it would be and responded to the training data processing in unpredictable ways. 
The last important lesson I learned was to pickle your models. Pickling allows you to save a specific instance of a model so that when your kernel is reset between work sessions, you do not have to fit the model again in order to use it to predict and score different data on. 

Given the lack of safe and easy access to clean drinking water in Tanzania this is a dataset that can be used to affect real positive change in the world by finding contributing causes to poorly maintained well sites and predicting which well sites may need repair soon. I hope that I'm able to revisit this dataset at some point in the future to approach this problem from that perspective and offer actionable input. 

If I had more time with this project I would have liked to explore feature importances and ranking, PCA(Principal Component Analysis), and CatBoosting models. My score placed me in the top 20% of competition entry scores and given my level of experience and that this was my first classification project I am very pleased with the outcome.

