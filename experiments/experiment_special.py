import argparse
import os

import pandas as pd

import logging
import traceback

logging.basicConfig(filename='experiment_special.log', level=logging.INFO)

import sys
# sys.path.append('/home/k2252103/Desktop/2023_feature_generator')
sys.path.append('/home/soyoung/Desktop/2023_feature_generator')
# sys.path.append('/Users/soyoungpark/Desktop/7CCSMPRJ/2023_feature_generator')
from src.dr_classes_special import *
from experiments.utils.ml_tasks_special import *
from experiments.utils.dataset_utils import load_train_test, split_to_X_and_y

import warnings
warnings.filterwarnings("ignore")

import gc

def run_experiment(
        experiment_id,
        data_path,
        task,
        dr_techniques,
        output_dir
):
        try:
            # Log the experiment details
            info = "Starting experiment with experiment id: " + str(experiment_id)
            logging.info(info)

            # Load the datasets for the given task
            folder_path = f"{data_path}/{task}/datasets"
            dataset_info = []
            
            train_folder = os.path.join(folder_path, "train")
            test_folder = os.path.join(folder_path, "test")

            for root, dirs, files in os.walk(train_folder):
                for file in files:
                    # Get the full path of the CSV file
                    file_path = os.path.join(train_folder, file)
                    name_part = file.split(".")[0]

                    # Load the CSV file to get its size (number of rows)
                    df = pd.read_csv(file_path)
                    dataset_size = len(df)

                    dataset_info.append((name_part, dataset_size))

            # Sort the dataset_info by size in ascending order
            dataset_info.sort(key=lambda x: x[1])
            
            print(dataset_info)

            # experiment_df = pd.DataFrame()
            experiments_dir = os.path.join(output_dir, task, f"experiment_{experiment_id}")
            os.makedirs(experiments_dir, exist_ok=True)

            # Now you can iterate over the sorted dataset names
            for dr_technique_name, dr_technique in dr_techniques.items():
                print("----------" + dr_technique_name + "----------")

                for dataset_name, _ in dataset_info:

                    file_path = os.path.join(experiments_dir, f"{dataset_name}_{dr_technique_name.upper()}_results.csv")

                # Open the file in write mode to erase any existing content

                    with open(file_path, 'w') as file:
                        if task == "classification": header = "dr_name,dataset_name,n_components,n_neighbors,accuracy,precision,recall,f1_score\n"
                        else: header = "dr_name,dataset_name,n_components,n_neighbors,RMSE,MAE,MAE_percentage,R2,residuals_mean,residuals_var\n"
                        file.write(header)

                    # print(dataset_name)
                    # Define the file names for train and test
                    train_file = os.path.join(train_folder, f"{dataset_name}.csv")
                    test_file = os.path.join(test_folder, f"{dataset_name}.csv")

                    # Read the CSV files into pandas DataFrames
                    train_data = pd.read_csv(train_file)
                    test_data = pd.read_csv(test_file)
                            
                    # Split the data into features and labels
                    X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
                    X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

                    # Apply DR model                    
                    reduced_data = dr_technique.reduce(dataset_name, X_train, X_test)
                    
                    del X_train, X_test
                    gc.collect()

                    # Apply ML model
                    if task == "classification": ml_classification_reduced(reduced_data, y_train, y_test, file_path)
                    if task == "regression": ml_regression_reduced(reduced_data, y_train, y_test, file_path)

                    del y_train, y_test
                    gc.collect()

                    del reduced_data
                    gc.collect()

                # Log success message
                info = "Finished fitting " + dr_technique_name + " on " + task + " experiment id: " + str(experiment_id)
                logging.info(info)

            info = "Finished experiment with experiment id: " + str(experiment_id)
            logging.info(info)
        
        except Exception as e:
            full_traceback = traceback.format_exc()

            print(full_traceback)
            
            # Log the error message along with the full traceback
            logging.error(f"Error occurred: {e}\nFull Traceback:\n{full_traceback}")


dr_techniques = pd.Series({
    'pca': PCA(),
    'kpca': KPCA(),
    'ica': ICA(),
    'svd': SVD(),
    'lpp': LPP(),
    'umap': UMAP(),
    'isomap': ISOMAP(),
    'lle': LLE()})

# Set the output directory
OUTPUT_DIR = "/home/soyoung/Desktop/2023_feature_generator/experiments/results"

'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--task', type=str, required=True)
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('-id', '--experiment_id', type=int, required=True)

    script_arguments = parser.parse_args()

    print("Entered python experiment_script.py")

    run_experiment(
        script_arguments.experiment_id,
        script_arguments.data_path,
        script_arguments.task,
        dr_techniques,
        OUTPUT_DIR
    )

    print("Finished python experiment_script.py")
'''