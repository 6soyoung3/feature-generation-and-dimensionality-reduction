##########################################################################################
'''
Reduced Datasets ML training
'''
##########################################################################################

import traceback
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, precision_score, recall_score, r2_score

import gc

import sys
sys.path.append('../2023_feature_generator/experiments')
from utils.dataset_utils import *

import gc

def ml_classification(data_info, reduced_X_train, reduced_X_test, y_train, y_test, file_path):
    clf = svm.SVC(kernel='rbf')
    try:   
        if np.isnan(reduced_X_train).any():
            raise ValueError("Input X contains NaN values.")

        clf.fit(reduced_X_train, y_train)                 
                        
        # Make predictions on the reduced test data
        y_pred = clf.predict(reduced_X_test)

        del reduced_X_train, reduced_X_test
        gc.collect()

        # Determine the type of classification problem
        n_classes = len(np.unique(y_test))
        average_method = 'binary' if n_classes == 2 else 'weighted'

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=average_method)
        prec = precision_score(y_test, y_pred, average=average_method, zero_division=1)
        recall = recall_score(y_test, y_pred, average=average_method)

        # Append results to the dataframe
        result = pd.DataFrame({
                'dr_name': data_info['dr_name'],
                'dataset_name': data_info['dataset_name'],
                'n_components': data_info['n_components'],
                'n_neighbors': data_info['n_neighbors'],
                'accuracy': acc,
                'precision': prec,
                'recall': recall,
                'f1_score': f1
            }, index=[0])
            
        mode = 'a' if os.path.exists(file_path) else 'w'  # Append if file exists, else write
        result.to_csv(file_path, mode=mode, header=not os.path.exists(file_path), index=False)

        print(f"dataset_name: {result['dataset_name'].values} : n_components: {result['n_components'].values} : n_neighbors : {result['n_neighbors'].values}")
            
    except Exception as e:

            full_traceback = traceback.format_exc()

            print(full_traceback)
            
            print(e)
            print("dataset: ", data_info['dataset_name'])
            print("n_components: ", data_info['n_components'])
            print("n_neighbors: ", data_info['n_components'])

def ml_regression(data_info, reduced_X_train, reduced_X_test, y_train, y_test, file_path):
    reg = svm.SVR(kernel='rbf')
    try:   
        if np.isnan(reduced_X_train).any():
            raise ValueError("Input X contains NaN values.")

        reg.fit(reduced_X_train, y_train)                 
                        
        # Make predictions on the reduced test data
        y_pred = reg.predict(reduced_X_test)

        del reduced_X_train, reduced_X_test
        gc.collect()

        # Calculate metrics
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        mae_percentage = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        residuals = y_test - y_pred
        residuals_mean = residuals.mean()
        residuals_var = residuals.var()

        # Append results to the dataframe
        result = pd.DataFrame({
                'dr_name': data_info['dr_name'],
                'dataset_name': data_info['dataset_name'],
                'n_components': data_info['n_components'],
                'n_neighbors': data_info['n_neighbors'],
                'RMSE': rmse,
                'MAE': mae,
                'MAE_percentage': mae_percentage,
                'R2': r2,
                'residuals_mean': residuals_mean,
                'residuals_var': residuals_var
            }, index=[0])
            

        mode = 'a' if os.path.exists(file_path) else 'w'  # Append if file exists, else write
        result.to_csv(file_path, mode=mode, header=not os.path.exists(file_path), index=False)

        print(f"dataset_name: {result['dataset_name'].values} : n_components: {result['n_components'].values} : n_neighbors : {result['n_neighbors'].values}")
            
    except Exception as e:

            full_traceback = traceback.format_exc()

            print(full_traceback)
            
            print(e)
            print("dataset: ", data_info['dataset_name'])
            print("n_components: ", data_info['n_components'])
            print("n_neighbors: ", data_info['n_components'])

def ml_classification_reduced(reduced_data, y_train, y_test, file_path):

    clf = svm.SVC(kernel='rbf')
    
    for index, row in reduced_data.iterrows():
        try: 
            reduced_X_train = row['reduced_X_train']
            reduced_X_test = row['reduced_X_test']

            if np.isnan(reduced_X_train).any():
                raise ValueError("Input X contains NaN values.")

            clf.fit(reduced_X_train, y_train)                 
                        
            # Make predictions on the reduced test data
            y_pred = clf.predict(reduced_X_test)

            del reduced_X_train, reduced_X_test
            gc.collect()

            # Determine the type of classification problem
            n_classes = len(np.unique(y_test))
            average_method = 'binary' if n_classes == 2 else 'weighted'

            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average=average_method)
            prec = precision_score(y_test, y_pred, average=average_method, zero_division=1)
            recall = recall_score(y_test, y_pred, average=average_method)

            # Append results to the dataframe
            result = pd.DataFrame({
                    'dr_name': row['dr_name'],
                    'dataset_name': row['dataset_name'],
                    'n_components': row['n_components'],
                    'n_neighbors': row['n_neighbors'],
                    'accuracy': acc,
                    'precision': prec,
                    'recall': recall,
                    'f1_score': f1
                }, index=[0])
            

            mode = 'a' if os.path.exists(file_path) else 'w'  # Append if file exists, else write
            result.to_csv(file_path, mode=mode, header=not os.path.exists(file_path), index=False)

            print(f"dataset_name: {result['dataset_name']} : n_components: {result['n_components']} : n_neighbors : {result['n_neighbors']}")
            
        except Exception as e:

            full_traceback = traceback.format_exc()

            print(full_traceback)
            
            print(e)
            print("dataset: ", row['dataset_name'])
            print("n_components: ", row['n_components'])
            print("n_neighbors: ", row['n_components'])

def ml_regression_reduced(reduced_data, y_train, y_test, file_path):

    reg = svm.SVR(kernel='rbf')

    for index, row in reduced_data.iterrows():
        try: 
            reduced_X_train = row['reduced_X_train']
            reduced_X_test = row['reduced_X_test']

            if np.isnan(reduced_X_train).any():
                raise ValueError("Input X contains NaN values.")

            reg.fit(reduced_X_train, y_train)

            # Make predictions on the reduced test data
            y_pred = reg.predict(reduced_X_test)

            del reduced_X_train, reduced_X_test
            gc.collect()

            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            mae_percentage = mean_absolute_percentage_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            residuals = y_test - y_pred
            residuals_mean = residuals.mean()
            residuals_var = residuals.var()

            # Append results to the dataframe
            result = pd.DataFrame({
                'dr_name': row['dr_name'],
                'dataset_name': row['dataset_name'],
                'n_components': row['n_components'],
                'n_neighbors': row['n_neighbors'],
                'RMSE': rmse,
                'MAE': mae,
                'MAE_percentage': mae_percentage,
                'R2': r2,
                'residuals_mean': residuals_mean,
                'residuals_var': residuals_var
            }, index=[0])
            
            result.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)

            print(f"dataset_name: {result['dataset_name']} : n_components: {result['n_components']} : n_neighbors : {result['n_neighbors']}")
            
        except Exception as e:
            print(e)
            print("dataset: ", row['dataset_name'])
            print("n_components: ", row['n_components'])
            print("n_neighbors: ", row['n_components'])