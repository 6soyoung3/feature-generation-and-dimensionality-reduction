import argparse

import sys
sys.path.append('/home/soyoung/Desktop/2023_feature_generator')
from src.dr_classes import *
from experiments.utils.ml_tasks import *
from experiments.utils.dataset_utils import load_train_test_dir, split_to_X_and_y

# Set the output directory
OUTPUT_DIR = "/home/soyoung/Desktop/2023_feature_generator/experiments/results"

def run_ml_reference(task, data_path):
    output_path = f"{OUTPUT_DIR}/{task}/reference_results.csv"
    folder_path = f"{data_path}/{task}/datasets"

    with open(output_path, 'w') as file:
        if task == "classification": header = "dataset_name,accuracy,precision,recall,f1_score\n"
        else: header = "dataset_name,RMSE,MAE,MAE_percentage,R2,residuals_mean,residuals_var\n"
        file.write(header)

    # Load original datasets from the given folder
    train_folder, test_folder, dataset_info = load_train_test_dir(folder_path)

    for dataset_name in dataset_info:
        # Define the file names for train and test
        train_file = load_file(train_folder, dataset_name)
        test_file = load_file(test_folder, dataset_name)

        # Read the CSV files into pandas DataFrames
        train_data, test_data = pd.read_csv(train_file), pd.read_csv(test_file)
                            
        # Split the data into features and labels
        X_train, y_train = split_to_X_and_y(train_data)
        X_test, y_test = split_to_X_and_y(test_data)

        if task == "classification":
            ml_classification_reference(dataset_name, X_train, y_train, X_test, y_test, output_path)
        else: ml_regression_reference(dataset_name, X_train, y_train, X_test, y_test, output_path)

'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--task', type=str, required=True)
    parser.add_argument('-d', '--data_path', type=str, required=True)

    script_arguments = parser.parse_args()

    print("Entered python ml_reference_script.py")

    run_ml_reference(
        script_arguments.task,
        script_arguments.data_path,
    )

    print("Finished python ml_reference_script.py")
'''

run_ml_reference("classification", '/home/soyoung/Desktop/2023_feature_generator/data')
run_ml_reference("regression", '/home/soyoung/Desktop/2023_feature_generator/data')