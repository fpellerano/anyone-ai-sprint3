# -*- coding: utf-8 -*-
"""Copia de AnyoneAI - Project 3.ipynb

impute:
    para inputar tenemos que tomar el conjunto de train, fitear y aplicar pero todo en train.
    en datos nonumericos, fit >> el valor que mas se repite

test - out no - 


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
from sklearn import preprocessing
import time


"""
We need to predict whether 0 people qualify to get a loan, or 1 not.

"""
warnings.filterwarnings('ignore')
application_train_df = pd.read_csv('C:/Users/u189197/Desktop/TAMBO/AnyoneAI/Sprint3/dataset/application_train.csv', index_col=0)
application_test_df = pd.read_csv('C:/Users/u189197/Desktop/TAMBO/AnyoneAI/Sprint3/dataset/application_test.csv', index_col=0)
target=application_train_df.loc[:,['TARGET']]

application_train = application_train_df.copy()
application_test = application_test_df.copy()

application_train.drop(['TARGET'], inplace=True, axis=1)
features = pd.DataFrame(application_train.nunique()) #cuenta todos los valores unicos por feature
application_train.replace('XNA', np.nan, inplace=True)
application_train.replace('XAP', np.nan, inplace=True)
application_test.replace('XNA', np.nan, inplace=True)
application_test.replace('XAP', np.nan, inplace=True)
application_train['DAYS_EMPLOYED'].replace(365243, np.nan, inplace = True)
application_test['DAYS_EMPLOYED'].replace(365243, np.nan, inplace = True)

""" OUTLIERS train """
# sns.boxplot(data=application_train,y='DAYS_EMPLOYED').set(title='Outlayer Salary')
# DAYS_EMPLOYED positivos llevar a na 
# los bigotes de mayoy o menor llevarlos a la mediana 
#pre
# sns.boxplot(data=application_train, y="DAYS_EMPLOYED").set(title='Outlayer Salary')
# plt.figure(figsize=(16, 8))
# sns.displot(application_train, x='DAYS_EMPLOYED', kind='kde')

col_outliers =  application_train.select_dtypes(include='number').columns.tolist()
for feature in col_outliers:
    Q1 = application_train[feature].quantile(0.25)
    Q3 = application_train[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - (1.5*IQR)
    upper = Q3 + (1.5*IQR)
    # print(f'The numerical_feature {feature} has lower: {lower} upper: {upper} and range IQR: {IQR}')
    application_train.loc[application_train[feature]>upper,feature]=application_train[feature].median()
    application_train.loc[application_train[feature]<lower,feature]=application_train[feature].median()
    # print(b)
#post
# sns.boxplot(data=application_train, y="DAYS_EMPLOYED").set(title='Outlayer Salary')
# plt.figure(figsize=(16, 8))
# sns.displot(application_train, x='DAYS_EMPLOYED', kind='kde')


""" División de numerical features"""
# columns of numerical features
numerical_features = pd.DataFrame(application_train.select_dtypes(exclude='object').columns)
# sum of features unique of numerical features
numerical_features = numerical_features.merge(features, left_on=0, right_index=True)
numerical_features = numerical_features.rename(columns={'key_0':'0','0_y':'1'})
numerical_features = numerical_features.loc[:,['0','1']]
# data of n.f. 
numerical_features = application_train.loc[:,(numerical_features['0'].tolist())]
# PRE
# percent of nan row of n.f.
# nan_values_numerical_features = (numerical_features.isnull().sum()/numerical_features.shape[0]*100).sort_values(ascending=False)
# sns.displot(numerical_features, x='COMMONAREA_AVG', kind='kde')
# FIT: completar los NAN con la mediana 
numerical_features = numerical_features.fillna(numerical_features.median())
# percent of nan row of n.f pos fillna.
# POST
# sns.displot(numerical_features, x='COMMONAREA_AVG', kind='kde')
# nan_values_numerical_features_2 = (numerical_features.isnull().sum()/numerical_features.shape[0]*100).sort_values(ascending=False)
numerical_features_t = pd.DataFrame(application_test.select_dtypes(exclude='object').columns)
numerical_features_t = numerical_features_t.merge(features, left_on=0, right_index=True)
numerical_features_t = numerical_features_t.rename(columns={'key_0':'0','0_y':'1'})
numerical_features_t = numerical_features_t.loc[:,['0','1']]
numerical_features_t = application_test.loc[:,(numerical_features_t['0'].tolist())]
numerical_features_t = numerical_features_t.fillna(numerical_features_t.median())


""" División de categorical features"""
string_features = pd.DataFrame(application_train.select_dtypes(include='object').columns)
string_features = string_features.merge(features, left_on=0, right_index=True)
string_features = string_features.rename(columns={'key_0':'0','0_y':'1'})
string_features = string_features.loc[:,['0','1']]
string_features = application_train.loc[:,(string_features['0'].tolist())]
# nan_values_categorial_features = (string_features.isnull().sum()/string_features.shape[0]*100).sort_values(ascending=False)
string_features = string_features.apply(lambda x: x.fillna(x.value_counts().index[0]))
# nan_values_categorial_features_2 = (string_features.isnull().sum()/string_features.shape[0]*100).sort_values(ascending=False)
string_features_t = pd.DataFrame(application_test.select_dtypes(include='object').columns)
string_features_t= string_features_t.merge(features, left_on=0, right_index=True)
string_features_t= string_features_t.rename(columns={'key_0':'0','0_y':'1'})
string_features_t= string_features_t.loc[:,['0','1']]
string_features_t= application_test.loc[:,(string_features_t['0'].tolist())]
# nan_values_categorial_features = (string_features.isnull().sum()/string_features.shape[0]*100).sort_values(ascending=False)
string_features_t= string_features_t.apply(lambda x: x.fillna(x.value_counts().index[0]))
# nan_values_categorial_features_2 = (string_features.isnull().sum()/string_features.shape[0]*100).sort_values(ascending=False)

# """ INPUTE test (fit_transform) / test (transform)
#     ohe categorical features with > 3
#     LabelEncoder()  binarias y drop first
#     StandartScaler() para los continuos

 # """

application_train.update(numerical_features, join='left', overwrite=True)
application_train.update(string_features, join='left', overwrite=True)


application_test.update(numerical_features_t, join='left', overwrite=True)
application_test.update(string_features_t, join='left', overwrite=True)

#237

def get_dummies(train, test):
    lbl_encoder = preprocessing.LabelEncoder()
    oh_encoder = OneHotEncoder(dtype=int, drop='first', sparse=False)
    train_dummies = pd.DataFrame()
    test_dummies = pd.DataFrame()
    for col in application_train:
        if application_train[col].dtype == 'object':
            if len(list(application_train[col].unique())) < 3:
                lbl_encoder.fit(application_train[col])
                application_train[col] = lbl_encoder.transform(application_train[col])
                application_test[col] = lbl_encoder.transform(application_test[col])
            else:
                train_dummies = oh_encoder.fit_transform(application_train[[col]])
                application_train[oh_encoder.categories_[0][1:]] = train_dummies
                application_train.drop(col, axis=1, inplace=True)
                test_dummies = oh_encoder.transform(application_test[[col]])
                application_test[oh_encoder.categories_[0][1:]] = test_dummies
                application_test.drop(col, axis=1, inplace=True)
    train_col_list = list(application_train)

dummies = get_dummies(application_train, application_test)   
# features_2 = pd.DataFrame(application_train.nunique()) #cuent

def scaler_mm(train, test):
    numerical_features = application_train.select_dtypes(include='number').columns
    for i in numerical_features:
        if len(list(application_train[i].unique())) > 2:
            scaler_mm = MinMaxScaler()
            application_train[i] = scaler_mm.fit_transform(application_train[i].values.reshape(-1,1)) # se escala la o las variables independiente, NO las dependientes
            application_test[i] = scaler_mm.transform(application_test[i].values.reshape(-1,1)) # se escala la o las variables independiente, NO las dependientes
        
scalermm = scaler_mm (application_train, application_test)    

""" X train """
application_train = np.array(application_train)
""" X test """
application_test = np.array(application_test)
""" y train """
target= np.array(target)


""" MODEL """


# lreg = LogisticRegression()
# lreg.fit(application_train, target)
# lreg_pred = lreg.predict_proba(application_test)[:,1]
# print(f"Base Line Model Predict_proba: {lreg_pred}")

# df_kaggle = pd.DataFrame()
# df_kaggle['SK_ID_CURR'] = application_test_df.index
# df_kaggle['TARGET'] = lreg_pred

# df_kaggle.to_csv('C:/Users/u189197/Desktop/TAMBO/AnyoneAI/Sprint3/dataset/df_pellerano.csv', index=False)

# """
# Score: 0.72128
# Public score: 0.72746
# """


# train, test = get_dummies(train, test)    
#     # We make sure both dataframes have the same rows
#     for col in train_col_list:
#         if not col in list(test):
#             test[col] = 0
#     # We make sure rows are in the same order
#     test = test[train_col_list]
#     return application_train, test

# """ 
# LogisticRegression: 
# Score: 0.72128 
# Public score: 0.72746 
# """

# =======================================1==========================================#

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# start = time.time()
# rnd_forest_cl = RandomForestClassifier(random_state=7, n_jobs=-2)
# rnd_forest_cl.fit(application_train, target)
# rnd_forest_cl_predicts = rnd_forest_cl.predict_proba(application_test)[:,1]
# print(f"Base Line Model Predict_proba: {rnd_forest_cl_predicts}")

# end = time.time()
# print("Time took by Random Forest Fit: ", (end - start))
# df_kaggle = pd.DataFrame()
# df_kaggle['SK_ID_CURR'] = application_test_df.index
# df_kaggle['TARGET'] = rnd_forest_cl_predicts
# df_kaggle.to_csv('C:/Users/u189197/Desktop/TAMBO/AnyoneAI/Sprint3/dataset/df_pellerano_rnd_forest_cl.csv', index=False)

# """
# RandomForestClassifier
# Score: 0.63643
# Public score: 0.64663
# """

# =======================================1==========================================#

# from sklearn.model_selection import RandomizedSearchCV
# import scipy as sp

# hyperparameter_grid = {
#   'bootstrap': [False],
#   'max_depth': [10, None],
#    'max_features': ['auto', 'sqrt'],
#    'min_samples_leaf': sp.stats.randint(5, 10),
#    'min_samples_split': sp.stats.randint(7, 11),
#    'n_estimators': sp.stats.randint(270, 310)
# }

# start = time.time()
# rnd_forest_cl = RandomForestClassifier(random_state=2905, n_jobs=-2)
# clf = RandomizedSearchCV(rnd_forest_cl, hyperparameter_grid, random_state=2905, cv=None, scoring='roc_auc', n_iter=10, verbose=10)
# search = clf.fit(application_train, target)
# end = time.time()
# print(search.best_params_)

# end = time.time()
# print("Time took by Random Forest Fit: ", end - start)

# rnd_forest_cl_predicts = rnd_forest_cl.predict_proba(application_test)[:,1]
# print(f"Base Line Model Predict_proba: {rnd_forest_cl_predicts}")

# df_kaggle = pd.DataFrame()
# df_kaggle['SK_ID_CURR'] = application_test_df.index
# df_kaggle['TARGET'] = rnd_forest_cl_predicts
# df_kaggle.to_csv('C:/Users/u189197/Desktop/TAMBO/AnyoneAI/Sprint3/dataset/df_pellerano_rnd_forest_cl_2.csv', index=False)



# #=============================Training LightGBM Model==============================#
import lightgbm as lgb
train_x, test_x, train_y, test_y = train_test_split(application_train, target, test_size = 0.33, random_state = 2905, stratify=target)

param_grid = {
    'boosting_type': ['gbdt', 'dart'],
    'num_leaves': list(range(20, 150)),
    'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)),
    'subsample_for_bin': list(range(20000, 300000, 20000)),
    'min_child_samples': list(range(20, 500, 5)),
    'reg_alpha': list(np.linspace(0, 1)),
    'reg_lambda': list(np.linspace(0, 1)),
    'colsample_bytree': list(np.linspace(0.6, 1, 10)),
    'subsample': list(np.linspace(0.5, 1, 100)),
    'is_unbalance': [True, False]
}
parameters = {'objective': 'binary',
              'metric' : 'auc',
              'is_unbalance' : 'true',
              'boosting' : 'gbdt',
              'num_leaves' : 63,
              'feature_fraction' : 0.5,
              'bagging_fraction' : 0.5,
              'bagging_freq' : 20,
              'learning_rate' : 0.01,
              'verbose' : -1
            }
start = time.time()
train_set = lgb.Dataset(data=train_x, label=train_y)
test_set = lgb.Dataset(data=test_x, label=test_y)

lgb_model = lgb.train(parameters, train_set, valid_sets=test_set, num_boost_round=5000, early_stopping_rounds=50)
# cv_results = lgb_model.cv(param_grid, train_set, num_boost_round=10000, nfold=5, early_stopping_rounds=100, metrics='auc', seed=42)
#print(cv_results)
end = time.time()
print("Time took by LightGBM: ", end - start)


# # #==================================================================================#

# lgb_model_predicts = lgb_model.predict(application_test)
# print(f"LightGBM Predict_proba: {lgb_model_predicts}")

# df_kaggle = pd.DataFrame()
# df_kaggle['SK_ID_CURR'] = application_test_df.index
# df_kaggle['TARGET'] = lgb_model_predicts
# df_kaggle.to_csv('C:/Users/u189197/Desktop/TAMBO/AnyoneAI/Sprint3/dataset/df_pellerano_lgb_model_predicts.csv', index=False)

# """
# Score: 0.73644
# Public score: 0.73891
# """


# # #==================================================================================#












""" align """
# sumar celdas de train luego del impute, y reemplazarlas por un 0

# def train_test_column_dif(train:pd.DataFrame, test:pd.DataFrame):
#     columns_train = list(train.columns)
#     columns_test = list(test.columns)

#     for column in columns_train:
#         if column not in columns_test:
#             test[column] = np.zeros(len(test))

#     for column in columns_test:
#         if column not in columns_train:
#             train[column] = np.zeros(len(train))
#     return train, test


# application_train, application_test = train_test_column_dif(application_train, application_test)








#sacar el targuet antes de modelo




# LazyClasifier

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