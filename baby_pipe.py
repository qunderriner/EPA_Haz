"""
baby pipeline for ML projects

"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import seaborn as sns
sns.set()

import sklearn
from sklearn import linear_model
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn import metrics
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn import svm
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, precision_recall_curve
from sklearn import ensemble 
from sklearn import neighbors
from sklearn.grid_search import ParameterGrid
from datetime import timedelta

MODELS = {
    'logistic_regression': linear_model.LogisticRegression(), 
}

PARAMS = {
    'logistic_regression': {'C': [1,10]}, 
}

def load_data(filename):
    """
    This function loads the dataset from a CSV 
    """
    df = pd.read_csv(filename)
    return df

def make_that_y(df):
    df["EVALUATION_START_DATE"] = pd.to_datetime(df["EVALUATION_START_DATE"], errors = 'coerce')
    df['FOUND_VIOLATION'] = df['FOUND_VIOLATION'].str.strip()
    df['predvar'] = np.where(df['FOUND_VIOLATION'] == "Y", 1, 0)
    return df 

def change_that_type(df, colname):
    df[colname] = (df[colname]).astype(int)
    return df 

def dummy(df, colname):
    """
    Takes a categorical variable and creates binary/dummy variables
    Inputs:
        df (dataframe)
        colname (str) name of column to make dummys  
    """
    dummies = pd.get_dummies(df[colname]).rename(columns=lambda x: colname + "_" + str(x))
    df = pd.concat([df, dummies], axis=1)
    df = df.drop([colname], axis=1)
    return df

def get_xy(df, response, features):

    """
    Create data arrays for the X and Y values needed to be plugged into the model
    Inputs:
        df (dataframe) - the dataframe 
        response (str - the y value for the model 
        features (list of strings) - the x values for the model 
    """ 
    y = df[response].to_numpy()
    X = df[features].to_numpy()
    return X, y



def temporal_train_test_split(df, date_col, freq='12MS'):
    """
    produce six month interval splits of data
    inputs:
        df: dataframe
        date_col: column that has dates
    returns:
        list of dates
    """
    """
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    dates = pd.date_range(start=min_date, end=max_date, freq=freq)[1:]
    train_start = min_date
    splits = []
    for i, d in enumerate(dates[:-1]):
        splits.append([[train_start, d], [d+timedelta(days=gap_days), dates[i+1]+timedelta(days=gap_days)]])
    splits.append([[train_start, dates[-1]], [dates[-1]+timedelta(days=gap_days), max_date+timedelta(days=gap_days)]])
    return splits
	"""
	test_df_filter = (df["EVALUATION_START_DATE"] > '2016-1-1') & (df["EVALUATION_START_DATE"] <= '2016-12-31')
    test_df = df[test_df_filter]
    test = ['2016-1-1','2016-12-31']

    train_df_filter = (df["EVALUATION_START_DATE"] > '2017-1-1') & (df["EVALUATION_START_DATE"] <= '2017-12-31')
    train_df = df[train_df_filter]
    train = ['2017-1-1','2017-12-31']

	return [train, test]

def precision_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    precision = precision_score(y_true, preds_at_k)
    return precision

def split_data_by_time(df, date_col, response, features):
    """
    create training/testing splits of data 
    """
    test_df_filter = (df["EVALUATION_START_DATE"] > '2016-1-1') & (df["EVALUATION_START_DATE"] <= '2016-12-31')
    test_df = df[test_df_filter]
    test = ['2016-1-1','2016-12-31']

    train_df_filter = (df["EVALUATION_START_DATE"] > '2017-1-1') & (df["EVALUATION_START_DATE"] <= '2017-12-31')
    train_df = df[train_df_filter]
    train = ['2017-1-1','2017-12-31']

    X_train, y_train = get_xy(train_df, response, features)
    X_test, y_test = get_xy(test_df, response, features)
    return X_train, X_test, y_train, y_test, train, test 

def calculate_precision_at_threshold(predicted_scores, true_labels, threshold):
    """
    calculatesd recall score
        inputs:
            predicted_scores
            true_labels
            threshold
    """
    pred_label = [1 if x > threshold else 0 for x in predicted_scores]
    _, false_positive, _, true_positives = confusion_matrix(true_labels, pred_label).ravel()
    return 1.0 * true_positives / (false_positive + true_positives)

def calculate_precision_at_threshold_multi(predicted_scores, true_labels, thresholds):
    """
    calculatesd precision score for multiple thresholds
      inputs:
        predicted_scores
        true_labels
    """  
    z = []
    for i in thresholds:
        z.append(calculate_precision_at_threshold(predicted_scores, true_labels, i))
    return z 

def run_the_models(data, models_to_run, date_col, response, features):


    """
    This runs models and produces evaluation output:
    inputs:
        data: dataframe with data
        models_to_run: list of models to run 
        responce: column name of y variable
        features: list of column names for model features 
    returns:
        dataframe 
    """
    thresholds = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
    precision_cols = ["precision_at_{}".format(str(x)) for x in thresholds]
    recall_cols = ["recall_at_{}".format(str(x)) for x in thresholds]
    cols = ['model',
            'parameters',
            'train_start',
            'train_end',
            'test_start',
            'test_end',
            'f1_score',
            'auc'] + precision_cols + recall_cols
    model_results = []
    X_train, X_test, y_train, y_test, train, test = split_data_by_time(data, date_col, response, features)
    #do discretations and shit here - write a fucntion that takes in x and y(do i need to procsss y? maybe not) and does the discretion 
    splits = temporal_train_test_split(data, 'date_posted', freq='6M')
    for train, test in splits:
        X_train, X_test, y_train, y_test = split_data_by_time(data, 'date_posted', train, test, response, features)
        #do discretations and shit here - write a fucntion that takes in x and y(do i need to procsss y? maybe not) and does the discretion 
        X_train = 
        X_train = PROCESSS 

        for m in models_to_run:
            if m not in MODELS:
                print(m, 'bad model')
                break
            clf = MODELS[m]
            parameter_grid = ParameterGrid(PARAMS[m])
            for p in parameter_grid:
                try:
                    # initialize list to keep track of results
                    res = [m, p, train[0], train[1], test[0], test[1]]
                    clf.set_params(**p)
                    clf.fit(X_train, y_train)
                    predicted_scores = clf.predict_proba(X_test)[:,1]
                    predicted_vals = clf.predict(X_test)
                    true_labels = y_test
                    precise = calculate_precision_at_threshold_multi(predicted_scores, true_labels, thresholds)
                    recall = calculate_recall_at_threshold_multi(predicted_scores, true_labels, thresholds)
                    auc = sklearn.metrics.roc_auc_score(true_labels, predicted_vals)
                    f1 = sklearn.metrics.f1_score(true_labels, predicted_vals)
                    # append metrics to list
                    res = res + [auc, f1] + precise + recall 
                    model_results.append(res)
                except Exception as e:
                    print(e, m, p)
        df = pd.DataFrame(model_results, columns = cols)
    return df    









"""
    splits = temporal_train_test_split(data, 'date_posted', freq='6M')
    for m in models_to_run:
        if m not in MODELS:
            print(m, 'bad model')
            break
        clf = MODELS[m]
        parameter_grid = ParameterGrid(PARAMS[m]) 
        for p in parameter_grid:
            try:
                # initialize list to keep track of results
                res = [m, p, train[0], train[1], test[0], test[1]]
                clf.set_params(**p)
                clf.fit(X_train, y_train)
                predicted_scores = clf.predict_proba(X_test)[:,1]
                predicted_vals = clf.predict(X_test)
                true_labels = y_test
                precise = calculate_precision_at_threshold_multi(predicted_scores, true_labels, thresholds)
                model_results.append(res)
            except Exception as e:
                print(e, m, p)
    df = pd.DataFrame(model_results, columns=cols)
    return df
"""









