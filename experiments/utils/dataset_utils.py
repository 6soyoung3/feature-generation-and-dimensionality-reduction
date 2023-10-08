import os
import pandas as pd

# load original train and test files' paths
def load_train_test_dir(folder_path):
    dataset_info = []
            
    train_folder = os.path.join(folder_path, "train")
    test_folder = os.path.join(folder_path, "test")

    for root, dirs, files in os.walk(train_folder):
        for file in files:
            if file.endswith(".csv"):
                name_part = file.split(".")[0]              
                dataset_info.append(name_part)
    return train_folder, test_folder, dataset_info

# load the file
def load_file(path, name):
    return os.path.join(path, f"{name}.csv")

# split train and test files in to X_train, X_test, y_train and y_test
def split_to_X_and_y(df):
    return df.iloc[:, :-1], df.iloc[:, -1]

# generate meta data of the original datasets
def compute_and_save_data_stats(folder_path, summaries_dir, compute_stats):

    train_folder, test_folder, dataset_info = load_train_test_dir(folder_path)
    summary_df = pd.DataFrame()

    for dataset_name in dataset_info:
        # Define the file names for train and test
        train_file = load_file(train_folder, dataset_name)
        test_file = load_file(test_folder, dataset_name)

        # Read the CSV files into pandas DataFrames
        train_data, test_data = pd.read_csv(train_file), pd.read_csv(test_file)

        combined = pd.concat([train_data, test_data])
        stats = pd.DataFrame([compute_stats(combined)])
        transposed_stats = stats.T
        transposed_stats.index = transposed_stats.index.astype(str)

        # Add dataset_name as column name to the transposed DataFrame
        transposed_stats.columns = [dataset_name]
        
        # Append to the summary DataFrame
        summary_df = pd.concat([summary_df, transposed_stats], axis=1)

    # Convert summaries_dir to an absolute path
    abs_summaries_dir = os.path.abspath(summaries_dir)
    
    # If the directory doesn't exist, create it
    if not os.path.exists(abs_summaries_dir):
        os.makedirs(abs_summaries_dir)

    # Save the data_properties DataFrame to a CSV file in the summaries directory
    summary_df.to_csv(os.path.join(abs_summaries_dir, "meta_data.csv"), index=True)