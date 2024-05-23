'''
Python script for ML predicting ''predict.py'' written by Churna Bhandari on May 22, 2024
Utilities:
- load train module 
- import predict_models 
- Predict Hc for experimental materials
Author: Churna Bhandari
Date: May 22, 2024
Copyright (c) 2024 Churna Bhandari. This script is freely available for use and reproduction.
'''

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import pickle

# Import predict_models from train module
from train import predict_models

def write_predictions_to_excel(predictions_df, excel_filename):
    """
    Write predictions DataFrame to an Excel file with limited precision.
    Parameters:
        predictions_df (pandas.DataFrame): DataFrame containing predictions.
        excel_filename (str): Filename of the Excel file to write.
    Returns:
        None
    """
    with pd.ExcelWriter(excel_filename, engine="openpyxl") as writer:
        predictions_df.round(3).to_excel(writer, index=False, float_format="%.3f")

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # File directories
    EXPERIMENTAL_DATA_PATH = os.path.join(script_dir, "prediction", "experimental.xlsx")
    XGB_PREDICTION_FILE = os.path.join(script_dir, "prediction", "XGBprediction.xlsx")
    ANN_PREDICTION_FILE = os.path.join(script_dir, "prediction", "ANNprediction.xlsx")

    # Read experimental database
    df_exp = pd.read_excel(EXPERIMENTAL_DATA_PATH)
    Y_Hc_expt = df_exp['Hc(expt)']  # Target variable from experimental data
    X_exp = df_exp[['Ms', 'Aex', 'Ku']]  # Independent variables from experimental data
    X_exp_index = df_exp.set_index(['Compound'])  # Set compound column as index

    # Load trained models from file
    with open(os.path.join(script_dir, "trained_models.pkl"), 'rb') as f:
        models = pickle.load(f)

    # Generate predictions using trained models
    predictions = predict_models(models, np.log(X_exp))

    # Evaluate models only for test predictions
    test_keys = [key for key in predictions.keys()]
    #print(test_keys)

    # Extract predictions and format them
    Y_prediction_tunedxgb_df = pd.DataFrame(predictions['tuned_xgb_test'], columns=['Hc_predicted(T)'], index=Y_Hc_expt.index)
    Y_prediction_tunedxgb_df['Hc_original'] = Y_Hc_expt
    Y_prediction_tunedxgb_df.index = X_exp_index.index

    Y_predictedANN_exp_transform_df = pd.DataFrame(predictions['ann_test'], columns=['Hc_predicted(T)'], index=Y_Hc_expt.index)
    Y_predictedANN_exp_transform_df['Hc_original'] = Y_Hc_expt
    Y_predictedANN_exp_transform_df.index = X_exp_index.index

    # After loading the experimental data
   # print("Experimental DataFrame:")
    #print(df_exp.head())  # Check the contents of the experimental DataFrame

    # After generating predictions
    print("Predictions:")
    print(predictions)  # Check the contents of the predictions dictionary

    # Write predictions to Excel files
    write_predictions_to_excel(Y_prediction_tunedxgb_df, XGB_PREDICTION_FILE)
    write_predictions_to_excel(Y_predictedANN_exp_transform_df, ANN_PREDICTION_FILE)
