# -*- coding: utf-8 -*-
"""Copia de AnyoneAI - Project 3.ipynb

impute:
    para inputar tenemos que tomar el conjunto de train, fitear y aplicar pero todo en train.
    en datos nonumericos, fit >> el valor que mas se repite




### Getting the data

1- Login to Kaggle (if you don't have an account you'll have to register to get it) and download the [complete dataset](https://www.kaggle.com/competitions/home-credit-default-risk/data). Read the information about the data. What does a row in the main file represent? What does the target variable means?

One row represents one loan in our data sample.
The target variable says wether the loan was repaid (0) or not (1)

2- Load the training and test datasets, we're only going to work withe "application_train.csv" and "application_test.csv" for now
"""

# !pip install -q kaggle


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, f1_score, accuracy_score,\
    classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from pandas.plotting import scatter_matrix
import gzip
import warnings

"""
We need to predict whether 0 people qualify to get a loan, or 1 not.

"""
warnings.filterwarnings('ignore')
application_train = pd.read_csv('C:/Users/u189197/Desktop/TAMBO/AnyoneAI/Sprint3/dataset/application_train.csv', index_col=0)
#application_test = pd.read_csv('C:/Users/u189197/Desktop/TAMBO/AnyoneAI/Sprint3/dataset/application_test.csv', index_col=0)


# categorical_features = application_train.select_dtypes(include='object').columns
# categorical_features = application_train[categorical_features].nunique()
# no_categorical_features = application_train.select_dtypes(exclude='object').columns
# no_categorical_features = application_train[no_categorical_features].nunique()
features = pd.DataFrame(application_train.nunique()) #cuenta todos los valores unicos por feature
numerical_features = pd.DataFrame(application_train.select_dtypes(exclude='object').columns)
no_numerical_features = pd.DataFrame(application_train.select_dtypes(include='object').columns)
# features_ = pd.merge(features, numerical_features) 
# pd.merge(df1, df2, on="key", how="outer")
categorical_features = features[features[0]<4]
no_categorical_features = features[features[0]>4]

columns = pd.read_csv('C:/Users/u189197/Desktop/TAMBO/AnyoneAI/Sprint3/Project/columns.csv')
""" Training data """

# print(application_train.shape)
# app_train_head=application_train.head()
# app_train_describe=application_train.describe()
# app_train_types=application_train.dtypes

""" Missing values """

# print(pd.isnull(application_train['TARGET']).values.ravel().sum()) #sumo los valores nulos que hay en esa columna

# Borramos una columna si TODOS sus valores son nulos
# application_train.dropna(axis=0, how='all')

# Borramos una columna si ALGUNA sus valores son nulos
# application_train_2 = application_train.copy() 
# application_train_2 = application_train_2.dropna(axis=0, how='any')

# Impute de valores faltantes
# application_train_3 = application_train.copy() 
# application_train_3 = application_train_3['TARGET'].fillna(0) #completo con 0
# application_train_3 = application_train_3['TARGET'].fillna(application_train_3['TARGET'].mean()) #completo con la media
# application_train_3.fillna(method='ffill') #pone el valor cercano hacia adelante

""" Plots """
# % matplotlib inline
# savefig ("path_donde_guardar_im.jpeg)

# figure, axs = plt.subplots(2, 2, sharey=True, sharex=True)
# application_train.plot(kind='scatter', x='NAME_TYPE_SUITE', y='TARGET', ax=axs[0][0])
# application_train.plot(kind='scatter', x='NAME_INCOMF_TYPE', y='TARGET', ax=axs[0][1])
# application_train.plot(kind='scatter', x='OWN_CAR_AGE', y='TARGET', ax=axs[1][0])
# application_train.plot(kind='scatter', x='OCCUPATION_TYPE', y='TARGET', ax=axs[1][1])

# fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
# fig.suptitle('Initial Pokemon - 1st Generation')
# # Bulbasaur
# sns.barplot(ax=axes[0], x=bulbasaur.index, y=bulbasaur.values)
# axes[0].set_title(bulbasaur.name)
# # Charmander
# sns.barplot(ax=axes[1], x=charmander.index, y=charmander.values)
# axes[1].set_title(charmander.name)
# # Squirtle
# sns.barplot(ax=axes[2], x=squirtle.index, y=squirtle.values)
# axes[2].set_title(squirtle.name)

# sns.catplot(x='AMT_CREDIT' , kind='count', data=application_train)
# plt.xticks(rotation=90)


# plt.hist(application_train['TARGET'], bins = 20)
# plt.xlabel("Eje x")
# plt.ylabel("Eje y")
# plt.title("Título")

# plt.boxplot(application_train['TARGET']) 

# sns.boxplot(data=application_train, y="AMT_CREDIT").set(title='Outlayer Salary')
# plt.figure(figsize=(16, 8))
# sns.heatmap(application_train.corr(), annot=True)
""" Wrangling """
# loc
# valores numerical
# Valores no_numerical
# group
# CODE_GENDER = application_train.groupby['CODE_GENDER']


""" split """

   
# X_train = application_train.drop(['TARGET'], axis=1)
# y_train = application_train.loc[:,['TARGET']]
# print(f"Number of rows X_train: {len(X_train.axes[0])}")
# print(f"Number of columns X_train without TARGET: {len(X_train.axes[1])}")

# columns_numerical = X_train.select_dtypes(include='number').sum()
# columns_no_numerical = X_train.select_dtypes(exclude='number').count()
# print(columns_numerical)
# print(columns_numerical)

# sns.countplot(x='TARGET', data=application_train)
# app_train_values=application_train['TARGET'].value_counts()

"""
app_train['TARGET'].astype(int).plot.hist()
We have a imbalanced class problem.


"""


# def missing_data_plot(data, n=20):
#     missing_data(data).iloc[:n, 1].plot(kind='bar', title='{} largest missing fractions'.format(n))
# missing_data(application_train).head(10)
# for i in columns:
#     columns_[i] = columns 


# numerical_features_X_train = X_train.select_dtypes(include='number').columns.tolist()
# string_features_X_train = X_train.select_dtypes(exclude='number').columns.tolist()


# print(f"Number of seasons: {all_nba_df['season_id'].nunique()}")
# print(f"Number of players: {all_nba_df['player_id'].nunique()}")
# print(f"Number of all-nba selections: {len(all_nba_df[all_nba_df['all_nba']==1])}")
# print(f"Number of non-selected: {len(all_nba_df[all_nba_df['all_nba']==0])}")

# comp = pd.DataFrame({
#     'Overall': all_nba_df['all_nba'].value_counts() / len(all_nba_df),
#     'Stratified': y_test.value_counts() / len(y_test),
# }).sort_index()
# print(comp)


### Complete in this cell: Loading the dataset
#obtener los datos, loguear a kagle y bajarse el dataset entero
#cargar los dataset,app train y app test

"""### Exploratory Data Analysis

A lot of the analysis of the data can be found on public available Kaggle kernels or blog posts, but you need to make sure you understand the datasets properties before starting working on it, so we'll do exploratory data analysis for the main files

**Dataset Basics**

1- Show the shape of the training and test datasets.

###exploración de datdos
recomendación, investigar valores que hay en las columnas, valores nulos, overfeet
"""

### Complete in this cell: shape of the dataset

   


"""2- List all columns in the train dataset"""

### Complete in this cell: Show all columns in the training dataset 
#mostrar cuantas filas y columas tiee el dataset

"""3- Show the first 5 records of the training dataset, transpose the dataframe to see each record as a column and features as rows, make sure all features are visualized. Take your time to review what kind of information you can gather from this data."""

### Complete in this cell: show first 5 records in a transposed table
#para hacer un primer vistazo para que hay en esas columnas, si son binaras, datos continuoas discreotos, categóricoas.... etc...
#Cuando las var son categocias, a priori uno ya sabe el rango de opciónes que tenemos, asique al hacer de hacer OHE hay que balancear los valores.
#Ejemplo: tomamos la imputación para las columnas que tienen un 70% nan o vacíos.

"""4- Show the distribution of the target variable values: print the total value count and the percentage of each value, plot this relationship."""

### Complete in this cell: show distribution of target variable
#mostrar entre el pocentaje y los valores del traget.

"""5- Show the number of columns of each data type"""

### Complete in this cell: show number of columns per data type

"""6- For categorical variables, show the number of distinct values in each column (number of labels)"""

### Complete in this cell: show number of unique values per categorical column

"""7- Analyzing missing data: show the percentage of missing data for each column ordered by percentage descending (show only the 20 columns with higher missing pct)"""

### Complete in this cell: checking missing data
#ver si usamos ohe si lo usamos con varialbes ordinales o no. depende.

"""**Analyzing distribution of variables**

1- Show the distribution of credit amounts
"""

### Complete in this cell: distribution of credit amounts

"""2- Plot the education level of the credit applicants, show the percentages of each category. Also print the total counts for each category."""

### Complete in this cell: level of education plot

"""3- Plot the distribution of ocupation of the loan applicants"""

### Complete in this cell: ocupation of applicants

"""4- Plot the family status of the applicants"""

### Complete in this cell: family status

"""5- Plot the income type of applicants grouped by the target variable"""

### Complete in this cell: Income type of applicants by target variable

"""## Preprocessing

In this section, you will code a function to make all the data pre processing for the dataset. What you have to deliver is a function that takes the train and test dataframes, processes all features, and returns the transformed data as numpy arrays ready to be used for training.

The function should perform these activities:

- Correct outliers/anomalous values in numerical columns (hint: take a look at the DAYS_EMPLOYED column)
- Impute values for all columns with missing data (use median as imputing value)
- Encode categorical features:
    - If feature has 2 categories encode using binary encoding
    - More than 2 categories, use one hot encoding 
- Feature scaling

Keep in mind that you could get different number of columns in train and test because some category could only be present in one of the dataframes, this could create more one hot encoded columns. You should align train and test to have the same number of columns
"""

### Complete in this cell: Data pre processing function   
# se pide que hagamos una funión y que haga todos los pasos. 
# le paso el dataframe, se lo paso a la columnfuncion, y lista para aplicar al modelo.

"""## Training Models

As usual, you will start training simple models and will progressively move to more complex models and pipelines.

### Baseline: LogisticRegression

1- Import LogisticRegression from sklearn and train a model using the preprocesed train data from the previous section, and just default parameters. If you receive a warning because the algorithm failed to converge, try increasing the number of iterations or decreasing the C parameter
"""

### Complete in this cell: train a logistic regression 
# Hacer el baseline y subirlo a kaggle
# si está alrededor del 0.6 esta bien. 
# No enviar el resultado

"""2- Use the trained model to predict probabilites for the test data, and then save the results to a csv in the format expected in the competition: a SK_ID_CURR column and a TARGET column with probabilities. REMEMBER: the TARGET columns should ONLY contain the probabilities that the debt is not repaid (equivalent to the class 1)."""

### Complete in this cell: predict test data and save csv

"""3- Go to the Kaggle competition, and in the [submissions page](https://www.kaggle.com/competitions/home-credit-default-risk/submit) load your csv file. Report here the result in the private score you obtained.

At this point, the model should produce a result around 0.67

### Training a Random Forest Classifier

You're gonna start working in more complex models: ensambles, particularly, you're going to use the Random Forest Classifier from Scikit Learn.

1- Train a RandomForestClassifier, print the time taken by the fit function. Just use default hyperparameters, except for n_jobs, which should be set to "-1" to allow the library to use all CPU cores to speed up training time.
"""

### Complete in this cell: train a RandomForestClassifier
# entrenar un modelo

# Queremos que cuando les pasamos los datos, fit predict y nada mas..... igual que el anterior.

"""2- Use the classifier to predict probabilities on the test set, and save the results to a csv file."""

### Complete in this cell: predict test data and save csv

"""3- Load the predictions to the competition. Report the private score here."""

### Complete in this cell: report your score on Kaggle
# This model should have a private score around 0.68

"""### Randomized Search with Cross Validation

So far, we've only created models using the default hyperparameters of each algorithm. This is usually something that we would only do for baseline models, hyperparameter tuning is a very important part of the modeling process and is often the difference between having an acceptable model or not.

But, there are usually lots of hyperparameters to tune and a finite amount of time to do it, you have to consider the time and resources it takes to find an optimal combination of them. In the previous section you trained a random forest classifier and saw how much it took to train it once in your PC. If you want to do hyperparameter optimization you now have to consider that you will have to train the algorithm N number of times, with N being the cartesian product of all parameters. 

Furthermore, you can't validate the performance of your trained models on the test set, as this data should only be used to validate the final model. So we have to implement a validation strategy, K-Fold Cross Validation being the most common. But this also adds time complexity to our training, because we will have to train each combinations of hyperparameters M number of times, X being the number of folds in which we divided our dataset, so the total number of training iterations will be NxM... this resulting number can grow VERY quickly.

Fortunately there are strategies to mitigate this, here you're going to select a small number of hyperparameters to test a RandomForestClassifier, and use a Randomized Search algorithm with K-Fold Cross Validation to avoid doing a full search across the grid. 

Remember: take in consideration how much time it took to train a single classifier, and define the number of cross validations folds and iterations of the search accordingly. 
A recommendation: run the training process, go make yourself a cup of coffee, sit somewhere comfortably and forget about it for a while.

1- Use RandomizedSearchCV to find the best combination of hyperparameters for a RandomForestClassifier. The validation metric used to evaluate the models should be "roc_auc".
"""

### Complete in this cell: Use RandomizedSearchCV to find the best combination of hyperparameters for a RandomForestClassifier
# example_hyperparameter_grid = {
#  'bootstrap': [True, False],
#  'max_depth': [10, 50, 100, None],
#  'max_features': ['auto', 'sqrt'],
#  'min_samples_leaf': [1, 2, 4],
#  'min_samples_split': [2, 5, 10],
#  'n_estimators': [100, 200]
# }

# es una técnica para buscar hiperparámetros y correrlos.
# tener en cuenta que va a requerir mucho procesamiento de la pc...

"""2- Use the classifier to predict probabilities on the test set, and save the results to a csv file."""

### Complete in this cell: predict test data and save csv

"""3- Load the predictions to the competition. Report the private score here."""

### Complete in this cell: report your score on Kaggle
# This model should have a private score around 0.70

"""4- If you have the time and resources, you can train the model for longer iterations, or select more estimator sizes. This is optional, but if you, we would love to see your results.

### Optional: Training a LightGBM model

Gradient Boosting Machine is one of the most used machine learning algorithms for tabular data. Lots of competitions have been won using models from libraries like XGBoost or LightGBM. You can try using [LightGBM](https://lightgbm.readthedocs.io/en/latest/) to train a new model an see how it performs compared to the other classifiers you trained.
"""

### Complete in this cell: train a LightGBM model

"""### Optional: Using Scikit Learn Pipelines

So far you've created special functions or blocks or code to chain operations on data and then train the models. But, reproducibility is important, and you don't want to have to remember the correct steps to follow each time you have new data to train your models. There are a lots of tools out there that can help you with that, here you can use a [Sklearn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) to process your data.
"""

### Complete in this cell: use a sklearn Pipeline to automate the cleaning, standardizing and training