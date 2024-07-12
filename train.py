'''
Python script for ML training ''train.py'' written by Churna Bhandari on May 22, 2024
Utilities:
- ML on micromagnetic data obtained by solving Landau–Lifshitz–Gilbert equation with 
  mumax program (https://mumax.github.io/)
- Independent variables: Ms, Aex, Ku
- Target variable: Hc
- Evaluate model performance
- Predict Hc for experimental materials
Author: Churna Bhandari
Date: May 22, 2024
Copyright (c) 2024 Churna Bhandari. This script is freely available for use and reproduction.
'''

# Import necessary libraries
import os  # Operating system module for interacting with the operating system
import numpy as np  # Numerical computing library
import pandas as pd  # Data manipulation library
import xlsxwriter  # Library for writing Excel files
import csv
import pickle
import h5py
import random  # Library for generating random numbers
import warnings  # Library for handling warnings
import tensorflow as tf  # Deep learning library
#from tensorflow import keras  # High-level neural networks API
from tensorflow.keras import layers  # Layers for building neural networks
from tensorflow import keras  # High-level neural networks API
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # Metrics for evaluating models
from sklearn.model_selection import train_test_split  # Function for splitting dataset into train and test sets
from sklearn.linear_model import LinearRegression  # Linear regression model
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, ElasticNetCV  # Regularized regression models
from sklearn.tree import DecisionTreeRegressor  # Decision tree regressor model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # Ensemble models
from xgboost import XGBRegressor  # XGBoost regressor model
from statsmodels.stats.outliers_influence import variance_inflation_factor  # Function for calculating VIF

# Suppress warnings
warnings.filterwarnings('ignore')

# For GPU only
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)



# Set seeds to ensure reproducibility
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)  # Set environment variable for reproducibility
random.seed(seed_value)  # Set random seed for reproducibility
np.random.seed(seed_value)  # Set numpy random seed for reproducibility
tf.random.set_seed(seed_value)  # Set TensorFlow random seed for reproducibility

# Function to split data into train and test sets
def train_test_split(df, frac=0.3):
    # Splitting the dataframe into train and test sets
    test = df.sample(frac=frac, axis=0, random_state=43)  # Randomly sample test data
    train = df.drop(index=test.index)  # Drop test data from train set
    return train, test

# Function to perform log transformation
def log_transform(df, columns):
    # Applying log transformation to specified columns
    for col in columns[1:]:
        df[col] = np.log(df[col])
    return df

# Calculate VIF
def checking_vif(train):
    vif = pd.DataFrame()  # Create empty dataframe for VIF
    df = train.copy()  # Copy training data
    df.drop(['Compound', 'Hc'], axis='columns', inplace=True)  # Drop compound and Hc columns
    vif["feature"] = df.columns  # Set feature names
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]  # Calculate VIF
    return vif

def train_models(X_train, y_train):
    # Train Linear Regression Model
    model_lm = LinearRegression()
    model_lm.fit(X_train, y_train)

    # Train Lasso Model
    model_lasso = Lasso()
    model_lasso.fit(X_train, y_train)

    # Train LassoCV Model
    model_lasso_cv = LassoCV(alphas=[0.1], cv=11, n_jobs=-1)
    model_lasso_cv.fit(X_train, y_train)

    # Train Ridge Model
    model_ridge = Ridge()
    model_ridge.fit(X_train, y_train)

    # Train RidgeCV Model
    model_ridge_cv = RidgeCV(alphas=[0.1], cv=11)
    model_ridge_cv.fit(X_train, y_train)

    # Train ElasticNetCV Model
    model_elasticnet_cv = ElasticNetCV(alphas=[0.1], cv=11, n_jobs=-1)
    model_elasticnet_cv.fit(X_train, y_train)

    # Train Decision Tree Regressor
    dt = DecisionTreeRegressor()
    dt.fit(X_train, y_train)

    # Train Pruned Decision Tree Regressor
    dt_pruned = DecisionTreeRegressor(max_depth=5)
    dt_pruned.fit(X_train, y_train)

    # Train Random Forest Regressor
    rf = RandomForestRegressor(random_state=1)
    rf.fit(X_train, y_train)

    # Train Tuned Random Forest Regressor
    rf_tuned = RandomForestRegressor(n_estimators=1200, random_state=1, min_samples_leaf=3, bootstrap=True,
                                      max_features='sqrt', max_depth=5)
    rf_tuned.fit(X_train, y_train)

    # Train Gradient Boosting Regressor
    gbr = GradientBoostingRegressor(random_state=1)
    gbr.fit(X_train, y_train)

    # Train Tuned Gradient Boosting Regressor
    gbr_tuned = GradientBoostingRegressor(n_estimators=1200, min_weight_fraction_leaf=0, max_depth=5,
                                           learning_rate=0.007)
    gbr_tuned.fit(X_train, y_train)

    # Train XGBRegressor
    xgb = XGBRegressor(random_state=1, eval_metric='logloss')
    xgb.fit(X_train, y_train)

    # Train Tuned XGBRegressor for test dataset
    min_paramxgb = {"n_estimators": 1200, "max_depth": 5, "min_child_weight": 0,
                    "learning_rate": 0.007, "random_state": 42}
    tuned_xgb = XGBRegressor(**min_paramxgb, eval_metric='logloss')
    tuned_xgb.fit(X_train, y_train)

    # Train Tuned XGBRegressor for experimental dataset
    tuned_xgb_exp = XGBRegressor(**min_paramxgb, eval_metric='logloss')
    tuned_xgb_exp.fit(X_train, y_train)

    # Train ANN Model
    model = keras.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
        ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.007)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    model.fit(X_train.to_numpy(dtype=float), y_train.to_numpy(dtype=float), epochs=1500, batch_size=24, verbose=0)

    return {
        "model_lm": model_lm,
        "model_lasso": model_lasso,
        "model_lasso_cv": model_lasso_cv,
        "model_ridge": model_ridge,
        "model_ridge_cv": model_ridge_cv,
        "model_elasticnet_cv": model_elasticnet_cv,
        "dt": dt,
        "dt_pruned": dt_pruned,
        "rf": rf,
        "rf_tuned": rf_tuned,
        "gbr": gbr,
        "gbr_tuned": gbr_tuned,
        "xgb": xgb,
        "tuned_xgb": tuned_xgb,
        "ann": model
        }


def predict_models(models, X_test):
    # Model Prediction of Hc for test set using various models
    y_pred_dttest = np.exp(models['dt_pruned'].predict(X_test))
    y_pred_rftest = np.exp(models['rf_tuned'].predict(X_test))
    y_pred_gbrtest = np.exp(models['gbr_tuned'].predict(X_test))
    y_pred_xgbtest = np.exp(models['xgb'].predict(X_test))
    y_pred_tuned_xgbtest = np.exp(models['tuned_xgb'].predict(X_test))
    y_pred_ann_test = np.exp(models['ann'].predict(X_test.to_numpy(dtype=np.float32)))  # ANN prediction for test set

    # Create DataFrame for predictions
    predictions = {
        'dt_test': y_pred_dttest,
        'rf_test': y_pred_rftest,
        'gbr_test': y_pred_gbrtest,
        'xgb_test': y_pred_xgbtest,
        'tuned_xgb_test': y_pred_tuned_xgbtest,
        'ann_test': y_pred_ann_test
    }

    return predictions


def evaluate_models(y_true, predictions):
    if isinstance(predictions, np.ndarray):  # Check if predictions is a NumPy array
        r2 = r2_score(y_true, np.log(predictions))
        mae = mean_absolute_error(y_true, np.log(predictions))
        print("R-squared:", r2)
        print("Mean Absolute Error:", mae)
    else:
        for model_name, y_pred in predictions.items():
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            print(f"{model_name} R-squared:", r2)
            print(f"{model_name} Mean Absolute Error:", mae)
            print()

# Define function to write evaluation metrics to CSV file
def write_metrics_to_csv(metrics_file, metrics):
    """
    Write evaluation metrics to a CSV file.
    Parameters:
        metrics_file (str): Path to the CSV file.
        metrics (dict): Dictionary containing evaluation metrics for each model.
    Returns:
        None
    """
    with open(metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "R-squared", "Mean Absolute Error"])  # Write header row
        for model_name, (r2, mae) in metrics.items():
            writer.writerow([model_name, r2, mae])  # Write metrics to CSV file


if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # file direcotries
    MUMAX_DATA_PATH = os.path.join(script_dir, "micromagnetic", "mumax.csv")
    #MUMAX_DATA_PATH = r"D:\github\Gavin_Churna\magnetic_dataset\micromagnetic\mumax.csv"
    # Output files
    METRICS_FILE = "ml_metrics.csv"

    # Read micromagnetic simulated database
    df_mumax = pd.read_csv(MUMAX_DATA_PATH, names=['Ms', 'Aex', 'Ku', 'Hc'], header=None, na_values=['-1.0', '0.0']).dropna()

    # Create fictitious material ID for mumax dataset
    df_mumax['Compound'] = ['Id' + str(i) for i in range(len(df_mumax))]  # Generate compound IDs
    df_mumax = df_mumax[['Compound', 'Ms', 'Aex', 'Ku', 'Hc']]  # Reorder columns

    # Remove outliers
    df_mumax = df_mumax[(df_mumax['Hc'] >= 0.008) & (df_mumax['Hc'] <= 50)]  # Filter Hc values
    df_mumax['Hc'] = df_mumax['Hc'] * 0.25  # Adjust Hc values

    # Calculate quantiles and IQR
    Q1 = df_mumax['Hc'].quantile(0.25)  # First quartile
    Q3 = df_mumax['Hc'].quantile(0.75)  # Third quartile
    IQR = Q3 - Q1  # Interquartile range
    upper = Q3 + 1.5 * IQR  # Upper bound for outlier detection

    # Removing outliers
    df_mumax.drop(index=df_mumax[df_mumax.Hc > upper].index, inplace=True)  # Drop outlier rows

    #Compute the variation inflation factor of independent features in micromagnetic dataset
    print(checking_vif(df_mumax))

    # Split data into train and test sets
    train, test = train_test_split(df_mumax)

    # Log transformation
    columns = list(df_mumax.columns)  # Get list of column names
    train = log_transform(train, columns)  # Apply log transformation to train set
    test = log_transform(test, columns)  # Apply log transformation to test set
    columns = list(df_mumax.columns)  # Get updated list of column names

    x_cols = ['Ms', 'Aex', 'Ku']  # Exclude 'Compound' column from features
    X_train, X_test = train[x_cols], test[x_cols]
    y_train, y_test = train['Hc'], test['Hc']

    # Train models
    models = train_models(X_train, y_train)
    # Save trained models to a file
    with open('trained_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    # Generate predictions
    predictions = predict_models(models, X_test)
    #print("Available keys in predictions:", predictions.keys())

    # Evaluate models only for test predictions
    test_keys = [key for key in predictions.keys()]

    # Evaluate models for test dataset
    metrics = {}
    for model_name in test_keys:
        if model_name in predictions:
            r2, mae = None, None  # Initialize metrics
            if isinstance(predictions[model_name], np.ndarray):
                r2 = r2_score(y_test, np.log(predictions[model_name]))
                mae = mean_absolute_error(y_test, np.log(predictions[model_name]))
            else:
                y_pred = predictions[model_name]
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
            metrics[model_name] = (r2, mae)  # Store metrics in dictionary
    # Write evaluation metrics to CSV file
    write_metrics_to_csv(METRICS_FILE, metrics)

