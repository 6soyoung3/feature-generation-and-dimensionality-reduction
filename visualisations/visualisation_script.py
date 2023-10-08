import argparse

import sys
sys.path.append('/home/soyoung/Desktop/2023_feature_generator')
from experiments.utils.dr_visualisation import *

import os
import pandas as pd

##########################################################################################
'''
Load reference_results.csv 
'''
##########################################################################################

# Set the output directory
BASE_PATH = "/home/soyoung/Desktop/2023_feature_generator"
OUTPUT_DIR = BASE_PATH + "/experiments/results"
      
def merge_results(task):
    experiment_ids = list(range(1, 11))

    for id in experiment_ids:
        directory = f"experiment_{id}"

        csv_path = os.path.join(BASE_PATH, f"experiments/results/{task}", directory)
        output_file = os.path.join(csv_path, f"experiment_{id}.csv")

        if os.path.exists(output_file):
            os.remove(output_file)

        if task == "classification":
            columns = ['dr_name', 'dataset_name', 'n_components', 'n_neighbors', 'accuracy', 'precision', 'recall', 'f1_score']
        else:
            columns = ['dr_name', 'dataset_name', 'n_components', 'n_neighbors', 'RMSE', 'MAE', 'MAE_percentage', 'R2', 'residuals_mean', 'residuals_var']

        path_to_train_data = f"/home/soyoung/Desktop/2023_feature_generator/data/{task}/datasets/train"
        
        # Add the 'n_instances' column
        columns.append('n_instances')
        exp_df = pd.DataFrame(columns=columns)

        # Iterate over the CSV files in the directory
        for filename in os.listdir(csv_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(csv_path, filename)

                # Read CSV and skip the header
                temp_df = pd.read_csv(file_path, header=0)

                # Get the number of instances for each dataset in temp_df
                for index, row in temp_df.iterrows():
                    dataset_name = row['dataset_name']

                    dataset_path = os.path.join(BASE_PATH, path_to_train_data, f'{dataset_name}.csv') 
                    dataset = pd.read_csv(dataset_path)
                    num_instances = len(dataset)
                    temp_df.loc[index, 'n_instances'] = num_instances

                # Append the data to exp_df
                exp_df = exp_df.append(temp_df, ignore_index=True)

        # Save the merged results to a new CSV file
        exp_df.to_csv(output_file, index=False)

def load_experiment_outputs(task):
    # Hold all the dataframes loaded
    experiment_outputs = {}

    experiment_ids = list(range(1, 11))

    for id in experiment_ids:
        directory = f"experiment_{id}"
        
        csv_path = os.path.join(BASE_PATH, f"experiments/results/{task}", directory, f"experiment_{id}.csv")
        
        exp_df = pd.read_csv(csv_path)

        # Store the dataframe in a dictionary with directory name as key
        experiment_outputs[directory] = exp_df

    return experiment_outputs
            
def run_visualisation(task):
    # Load reference_results.csv
    ref_df = pd.read_csv(os.path.join(
                    BASE_PATH, f"experiments/results/{task}", "reference_results.csv"))
    
    # Create the visualisation directory for holding all the resulted plots (.png) if it doesn't exist
    results_path = os.path.join(BASE_PATH, "visualisations", task)
    os.makedirs(results_path, exist_ok=True)

    no_neighbors = ['PCA', 'ICA', 'SVD', 'KPCA']
    have_neighbors = ['LLE', 'LPP', 'ISOMAP', 'UMAP']

    if task == "classification":
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metrics_single = 'f1_score'

        metric_cmaps = {
            'accuracy': 'PuRd',
            'precision': 'Reds',
            'recall': 'Blues',
            'f1_score': 'Purples',
        }

        dataset_categories = {
            'High Dimensionality': ['Glass', 'Parkinsons', 'Zoo'],
            'Low Dimensionality': ['BuyCoupon', 'FakeWebsite', 'Grade', 'AcademicSuccess', 'StayHospital', 'Heart', 'DryBean'],
            'High Sparsity': ['BuyCoupon', 'StayHospital', 'Heart', 'Zoo'],
            'Low Sparsity': ['FakeWebsite', 'Grade', 'AcademicSuccess', 'Glass', 'Parkinsons', 'DryBean'],
            'High Redundancy': ['FakeWebsite', 'AcademicSuccess', 'Heart', 'Parkinsons', 'Zoo', 'DryBean'],
            'Low Redundancy': ['BuyCoupon', 'Grade', 'Glass', 'StayHospital'],
            'High Autocorrelation': ['FakeWebsite', 'Glass', 'StayHospital', 'Parkinsons', 'Zoo'],
            'Low Autocorrelation': ['BuyCoupon', 'Grade', 'AcademicSuccess', 'Heart', 'DryBean'],
        }


    else: 
        metrics = ['RMSE', 'MAE', 'MAE_percentage', 'R2', 'residuals_mean', 'residuals_var']
        metrics_single = 'RMSE'

        metric_cmaps = {
            'RMSE': 'Greens',
            'MAE': 'Reds',
            'MAE_percentage': 'Blues',
            'R2': 'Purples',
            'residuals_mean': 'coolwarm',
            'residuals_var': 'PuRd',
        }

        dataset_categories = {
            'High Dimensionality': ['HousePrice', 'Admission', 'ForestFires', 'PremiumPrice', 'Facebook', 'Credit'],
            'Low Dimensionality': ['WineQuality', 'NYCPropertySales', 'OnlineNewsPopularity', 'BankHappiness'],
            'High Redundancy': ['Admission'],
            'Low Redundancy': ['HousePrice', 'WineQuality', 'NYCPropertySales', 'ForestFires', 'PremiumPrice', 'OnlineNewsPopularity', 'BankHappiness', 'Facebook', 'Credit'],
            'High Sparsity': ['HousePrice', 'PremiumPrice', 'Credit'],
            'Low Sparsity': ['WineQuality', 'NYCPropertySales', 'ForestFires', 'OnlineNewsPopularity', 'BankHappiness', 'Facebook', 'Admission'],
            'High Autocorrelation': ['Admission', 'PremiumPrice', 'Facebook', 'Credit'],
            'Low Autocorrelation': ['HousePrice', 'WineQuality', 'NYCPropertySales', 'ForestFires', 'OnlineNewsPopularity', 'BankHappiness'],
        }

    experiment_ids = list(range(1,11))

    all_experiments_df = pd.concat([pd.read_csv(os.path.join(BASE_PATH, f"experiments/results/{task}", f"experiment_{id}", f"experiment_{id}.csv")) for id in experiment_ids], ignore_index=True)

    # Extract unique datasets
    datasets = set(all_experiments_df['dataset_name'].unique())

    # Call the functions
    # plot_lineplots(all_experiments_df, ref_df, metrics, datasets, results_path)     
    # plot_boxplots(all_experiments_df, ref_df, metrics, datasets, results_path)
    # plot_heatmaps(all_experiments_df, ref_df, metrics, datasets, metric_cmaps, results_path)

    # plot_aggregated_lineplot(all_experiments_df, metrics_single, results_path)
    # plot_aggregated_heatmap(all_experiments_df, metrics_single, results_path)
    # plot_aggregated_boxplot(all_experiments_df, metrics_single, results_path)
    # plot_boxplot_ref(all_experiments_df, ref_df, metrics_single, results_path)

    # plot_bubble(all_experiments_df, ref_df, metrics_single, results_path)
    plot_bubble_category(all_experiments_df, ref_df, metrics_single, dataset_categories, results_path)
    plot_box_category(all_experiments_df, ref_df, metrics_single, dataset_categories, results_path)

    plot_line_category(all_experiments_df, metrics_single, dataset_categories, results_path)

    # plot_combined_heatmap(all_experiments_df, metrics_single, results_path)    

    # plot_box_by_n_components_per_technique(all_experiments_df, ref_df, metrics_single, results_path, no_neighbors)
    # plot_box_by_n_components_per_technique(all_experiments_df, ref_df, metrics_single, results_path, have_neighbors)

    # rank_techniques(all_experiments_df, metrics_single, results_path)

    # count_improvements(all_experiments_df, ref_df, metrics_single, results_path)

'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--task', type=str, required=True)
    script_arguments = parser.parse_args()

    print("Entered python visualisation_script.py")
    
    run_visualisation(
        script_arguments.task,
        OUTPUT_DIR
    )

    print("Finished python visualisation_script.py")
'''

# merge_results("classification")
# merge_results("regression")

run_visualisation("classification")
run_visualisation("regression")
