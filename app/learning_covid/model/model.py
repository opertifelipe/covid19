from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
import pandas as pd
import numpy as np
import random

def rfr(X, y):
    """
    Random forest regression model
    Args:
        - X: features
        - y: target
    Return:
        Trained model

    """
    param_grid = {
        'rf__n_estimators': [100],
        'rf__max_depth': [2, 3, 4, 5, 6, 7, 8],
        'rf__min_samples_leaf': [3, 5]
    }
    pipeline = Pipeline([
        ('rf', RandomForestRegressor())
    ])

    CV_regr = GridSearchCV(pipeline, param_grid=param_grid, cv=5)
    CV_regr.fit(X, y)
    return CV_regr


def ridge(X, y):
    """
    Linear Ridge regression model
    Args:
        - X: features
        - y: target
    Return:
        Trained model

    """
    parameters = {'ridge__alpha': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  'ridge__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                  'ridge__fit_intercept': [True, False],
                  }
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge())
    ])

    CV_regr = GridSearchCV(pipeline, param_grid=parameters, cv=5)
    CV_regr.fit(X, y)
    return CV_regr


def lasso(X, y):
    """
    Linear Lasso regression model
    Args:
        - X: features
        - y: target
    Return:
        Trained model

    """
    parameters = {'lasso__alpha': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  'lasso__fit_intercept': [True, False],
                  'lasso__normalize': [True, False],
                  'lasso__positive': [True, False],
                  'lasso__selection': ['cyclic', 'random']
                  }
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', Lasso())
    ])

    CV_regr = GridSearchCV(pipeline, param_grid=parameters, cv=5)
    CV_regr.fit(X, y)
    return CV_regr


def kernel_ridge_regr(X, y):
    """
    Not-linear Ridge regression model
    Args:
        - X: features
        - y: target
    Return:
        Trained model

    """
    parameters = {'kr__alpha': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  'kr__gamma': [None, 0.1, 1, 2.0],
                  'kr__kernel': ['rbf', 'linear', 'polynomial'],
                  'kr__degree': [1, 2, 3, 4, 5]
                  }
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kr', KernelRidge())
    ])

    CV_regr = GridSearchCV(pipeline, param_grid=parameters, cv=5)
    CV_regr.fit(X, y)
    return CV_regr


def apply_model(model, X, y):
    """
    This function apply the model choosed
    Args:
        - model: model to apply
        - X: features
        - y: target
    Returns:
        Model trained
    """

    if model == "RFR":
        CV_regr = rfr(X, y)
    elif model == "ridge":
        CV_regr = ridge(X, y)
    elif model == "lasso":
        CV_regr = ridge(X, y)
    elif model == "kernel_ridge":
        CV_regr = kernel_ridge_regr(X, y)
    return CV_regr

def roll_predictions(day_to_roll, start_row, df_test, noise, CV_regr):
    """
    This function extend the prediction for several days after. It uses the result of the prediction for the
    next prediction
    Args:
        - day_to_roll: number of days to predict
        - start_row: day where we start the prediction
        - df_test: dataframe with the data
        - noise: additional noise in the result
        - CV_regr: trained model
    Returns:
        Dataframe with the the result of the prediction and the comparison
    """
    cols = list(df_test.columns)
    cols.remove("Target")
    cols.remove("Population")
    cols.remove("mean_age")
    cols.remove("GDP")
    cols.remove("Date")
    df_test_to_compare = df_test.copy()
    df_test = df_test.drop("Date", axis=1).reset_index(drop=True)
    row = df_test.iloc[start_row, 1:len(df_test.columns)].to_frame().transpose()
    preds = []
    for day in range(day_to_roll):
        prediction = CV_regr.predict(row)
        prediction = prediction[0] + random.uniform(0, 1) * noise
        preds.append(prediction)
        for i in range(len(cols) - 1):
            row[cols[i]] = row[cols[i + 1]]
        row[len(cols) - 1] = prediction

    predicted = pd.DataFrame(np.column_stack((list(np.arange(start_row, day_to_roll + start_row, 1)), preds)),
                             columns=["Day", "Target_predicted"])
    real = df_test_to_compare[["Target", "Date"]].reset_index(drop=False)
    real.columns = ["Day", "Target_real", "Date"]
    df_comparison = pd.merge(real, predicted, on="Day", how="outer")
    df_date_to_fill = pd.Series(
        pd.date_range(start=df_comparison["Date"].max(), periods=df_comparison["Date"].isnull().sum() + 1)).iloc[1:]
    df_exists = df_comparison[df_comparison["Date"].isnull() == False]["Date"]
    df_comparison["Date"] = df_exists.append(df_date_to_fill).values

    return df_comparison