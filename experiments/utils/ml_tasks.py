##########################################################################################
'''
Reduced Datasets ML training
'''
##########################################################################################

import gc
import traceback
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, precision_score, recall_score, r2_score

import sys
sys.path.append('../2023_feature_generator/experiments')
from utils.dataset_utils import *

# Mesaure ML performance using reference/origianl data (no DR applied)
def ml_classification_reference(dataset_name, X_train, y_train, X_test, y_test, file_path): 
    # Create a Support Vector Machine classifier
    clf = svm.SVC(kernel="rbf")

    # Calculate time taken to train the model and fit the classifier using the training data
    clf.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = clf.predict(X_test)

    # Determine the type of classification problem
    n_classes = len(np.unique(y_test))
    average_method = 'binary' if n_classes == 2 else 'weighted'

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=average_method)
    prec = precision_score(
        y_test, y_pred, average=average_method, zero_division=1)
    recall = recall_score(y_test, y_pred, average=average_method)

    # Create a new pandas Series to hold the results
    result = pd.DataFrame({'dataset_name': dataset_name,
                            'accuracy': acc,
                            'precision': prec,
                            'recall': recall,
                            'f1_score': f1
                            }, index=[0])
    
    mode = 'a' if os.path.exists(file_path) else 'w'  # Append if file exists, else write
    result.to_csv(file_path, mode=mode, header=not os.path.exists(file_path), index=False)

def ml_regression_reference(dataset_name, X_train, y_train, X_test, y_test, file_path):
    # Create a Support Vector Machine regressor
    reg = svm.SVR(kernel="rbf")

    # Calculate time taken to train the model and fit the regressor using the training data
    reg.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = reg.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    mae_percentage = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    residuals = y_test - y_pred
    residuals_mean = residuals.mean()
    residuals_var = residuals.var()

    # Create a new pandas Series to hold the results
    result = pd.DataFrame({'dataset_name': dataset_name,
                            'RMSE': rmse,
                            'MAE': mae,
                            'MAE_percentage': mae_percentage,
                            'R2': r2,
                            'residuals_mean': residuals_mean,
                            'residuals_var': residuals_var
                            }, index=[0])

    mode = 'a' if os.path.exists(file_path) else 'w'  # Append if file exists, else write
    result.to_csv(file_path, mode=mode, header=not os.path.exists(file_path), index=False)

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

def ml_classification_reduced(reduced_X_trains, reduced_X_tests, y_trains, y_tests, experiments_dir):
    reduced_results = pd.DataFrame(
        columns=['dr_name', 'dataset_name', 'n_components', 'n_neighbors', 'accuracy', 'precision', 'recall', 'f1_score'])
    
    print(reduced_results.columns)
        
    for reduced_data_name, reduced_X_train in reduced_X_trains.items():

        elements = reduced_data_name.split("_")
        dataset_name = elements.pop(0)
        dr_name = elements.pop(0)
        n_component = int(elements.pop(0))
        n_neighbor = None
        if elements:
            n_neighbor = int(elements.pop(0))

        y_train = y_trains[dataset_name]

        clf = svm.SVC(kernel='rbf')

        try: 
            # Time the training process
            clf.fit(reduced_X_train, y_train)

            # Fetch the corresponding reduced_X_test and y_test
            reduced_X_test = reduced_X_tests[reduced_data_name]
                    
            y_test = y_tests[dataset_name]
                    
            # Make predictions on the reduced test data
            y_pred = clf.predict(reduced_X_test)

            # Determine the type of classification problem
            n_classes = len(np.unique(y_test))
            average_method = 'binary' if n_classes == 2 else 'weighted'

            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average=average_method)
            prec = precision_score(y_test, y_pred, average=average_method, zero_division=1)
            recall = recall_score(y_test, y_pred, average=average_method)

            # Append results to the dataframe
            new_row = pd.DataFrame({
                'dr_name' : dr_name,
                'dataset_name': dataset_name,
                'n_components': n_component,
                'n_neighbors': n_neighbor,
                'accuracy': acc,
                'precision': prec,
                'recall': recall,
                'f1_score': f1
            }, index=[0])

            print(new_row.values)

            reduced_results = pd.concat(
                [reduced_results, new_row], ignore_index=True)
        except Exception as e:
            print(e)
            print(reduced_data_name)
            print(reduced_X_train.isnull().any)

    # Print the progress
    # print("Saving results for reduced_dataset_name: ", reduced_data_name)
    # Save the results DataFrame to a CSV file
    reduced_results.to_csv(experiments_dir + f"/{dr_name}_results.csv", index=False)
        
    return reduced_results

def ml_regression_reduced(reduced_X_trains, reduced_X_tests, y_trains, y_tests, experiments_dir):
    reduced_results = pd.DataFrame(
        columns=['dr_name', 'dataset_name', 'n_components', 'n_neighbors', 'RMSE', 'MAE', 'MAE_percentage', 'R2',
                 'residuals_mean', 'residuals_var'])
    
    print(reduced_results.columns)

    for reduced_data_name, reduced_X_train in reduced_X_trains.items():
        # Split the reduced_data_name to get the name, technique_name, n_component, and n_neighbor
        # e.g. BuyCoupon_LLE_2_9
        elements = reduced_data_name.split("_")
        dataset_name = elements.pop(0)
        dr_name = elements.pop(0)
        n_component = int(elements.pop(0))
        n_neighbor = None
        if elements:
            n_neighbor = int(elements.pop(0))

        # Fetch the corresponding y_train
        y_train = y_trains[dataset_name]

        reg = svm.SVR(kernel='rbf')

        try: 
            # Time the training process
            reg.fit(reduced_X_train, y_train)

            # Fetch the corresponding reduced_X_test and y_test
            reduced_X_test = reduced_X_tests[reduced_data_name]
                    
            y_test = y_tests[dataset_name]
                    
            # Make predictions on the reduced test data
            y_pred = reg.predict(reduced_X_test)

            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            mae_percentage = mean_absolute_percentage_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            residuals = y_test - y_pred
            residuals_mean = residuals.mean()
            residuals_var = residuals.var()

            # Calculate metrics
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            mae_percentage = mean_absolute_percentage_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            residuals = y_test - y_pred
            residuals_mean = residuals.mean()
            residuals_var = residuals.var()

            # Append results to the dataframe
            new_row = pd.DataFrame({
                'dr_name' : dr_name,
                'dataset_name': dataset_name,
                'n_components': n_component,
                'n_neighbors': n_neighbor,
                'RMSE': rmse,
                'MAE': mae,
                'MAE_percentage': mae_percentage,
                'R2': r2,
                'residuals_mean': residuals_mean,
                'residuals_var': residuals_var
                }, index=[0])

            print(new_row.values)

            reduced_results = pd.concat(
                [reduced_results, new_row], ignore_index=True)
            
        except Exception as e:
            print(e)
            print(reduced_data_name)
            # Detect the NaN values in the entire 2D array
            mask = np.isnan(reduced_X_train)

            # Identify the rows that contain at least one NaN value
            rows_with_nan = np.where(mask.any(axis=1))[0]

            # Print these rows
            for row in rows_with_nan:
                print(reduced_X_train[row])

    # Print the progress
    # print("Saving results for reduced_dataset_name: ", reduced_data_name)
    reduced_results.to_csv(experiments_dir + f"/{dr_name}_results.csv", index=False)

    return reduced_results