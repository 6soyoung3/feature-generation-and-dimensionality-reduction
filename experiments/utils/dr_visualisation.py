import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter

import warnings
warnings.filterwarnings("ignore")

#######################################################################################################
'''
Helper functions
'''
#######################################################################################################

def create_subplots(n_metrics, fig_size=(15, 5), n_cols=2):
    n_rows = n_metrics // n_cols + (n_metrics % n_cols > 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_size[0], n_rows * fig_size[1]))
    return fig, axes.flatten()

def save_plot(fig, path, file_name):
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, file_name))
    plt.close(fig)

def format_value(value):
    if value >= 1e9:
        return f"{value / 1e9:.0f}B"
    elif value >= 1e6:
        return f"{value / 1e6:.0f}M"
    elif value >= 1e3:
        return f"{value / 1e3:.0f}K"
    return f"{value:.3f}"

#######################################################################################################
'''
Plotting functions
'''
#######################################################################################################    

def plot_lineplots(exp_df, ref_df, metrics, datasets, results_path):
    sns.set_palette(sns.color_palette("husl", 8)) # Set the color palette
    for dataset in datasets:
        df_dataset = exp_df[exp_df['dataset_name'] == dataset]

        n_metrics = len(metrics)
        fig, axes = create_subplots(n_metrics)

        for i, metric in enumerate(metrics):
            ax = axes[i]            

            if df_dataset['n_neighbors'].isna().any():
                grouped = df_dataset.groupby(['dr_name', 'n_components'], as_index=False).mean()
            else:
                grouped = df_dataset.groupby(['dr_name', 'n_components', 'n_neighbors'], as_index=False).mean()

            unique_n_components = grouped['n_components'].unique()
            unique = len(unique_n_components) == 1
            if unique:
                n_component_value = unique_n_components[0]
                sns.barplot(data=grouped, x='dr_name', y=metric, ax=ax)
                ax.set_title(f'Bar plot of {metric} for {dataset} (n_components={int(n_component_value)})')
                ax.set_xlabel('techniques')
            else:
                sns.lineplot(data=grouped, x='n_components', y=metric, hue='dr_name', ax=ax)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensures that all ticks are integers
                ax.set_title(f'Line plot of {metric} for {dataset}')
                ax.set_xlabel('n_components')
                ax.legend(title='Techniques')

            ax.set_ylabel(metric.capitalize())

            ref_value = ref_df.loc[ref_df['dataset_name'] == dataset, metric].values[0]
            ref_value = round(ref_value, 3)  # Round to 3 decimal places
            ax.axhline(ref_value, color='r', linestyle='--')

            ref_x_position = max(unique_n_components)*0.9
            y_min, y_max = ax.get_ylim()
            ref_y_position = ref_value + 0.02 * (y_max - y_min)
            ax.text(ref_x_position, ref_y_position, f'Reference {metric}: {ref_value}', color='r',  ha='center')
            

        # Remove unused subplots
        if n_metrics < len(axes):
            for i in range(n_metrics, len(axes)):
                fig.delaxes(axes[i])

        fig.tight_layout()

        dir_path = os.path.join(results_path, 'line_plots')
        os.makedirs(dir_path, exist_ok=True)  # Check if the directory exists and create if not

        if unique:
            save_plot(fig, dir_path, f'{dataset}_barplot.png')
        else:
            save_plot(fig, dir_path, f'{dataset}_lineplot.png')

def plot_boxplots(exp_df, ref_df, metrics, datasets, results_path):
    sns.set_palette(sns.color_palette("husl", 8)) # Set the color palette
    for dataset in datasets:
        df_dataset = exp_df[exp_df['dataset_name'] == dataset]

        # Calculate the number of rows and columns for subplots
        n_metrics = len(metrics)
        
        # Create subplots using the helper function
        fig, axes = create_subplots(n_metrics)

        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Initialise an empty DataFrame for the metric
            df_metric = pd.DataFrame()

            # Plot the reference data
            ref_value = ref_df.loc[ref_df['dataset_name'] == dataset, metric].values[0]
            ref_value = round(ref_value, 3)  # Round to 3 decimal places
            ax.axhline(ref_value, color='r', linestyle='--')

            ax.text(0.5, ref_value, f'Reference {metric}: {ref_value}', color='r')
            
            # Select the metric and technique (or dr_name) from the dataset
            df_metric[metric] = df_dataset[metric]
            df_metric['Techniques'] = df_dataset['dr_name']

            # Create a box plot
            sns.boxplot(data=df_metric, x='Techniques', y=metric, ax=ax)
            ax.set_title(f'Box plot of {metric} for {dataset}')
            ax.set_ylabel(metric.capitalize())
            ax.grid(True)  # Show grid lines

        # Turn off the remaining axes that are not used
        for j in range(n_metrics, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        
        # Save the figure using the helper function
        dir_path = os.path.join(results_path, 'box_plots')
        os.makedirs(dir_path, exist_ok=True)  # Check if the directory exists and create if not

        file_name = f"{dataset}_boxplot.png"
        save_plot(fig, dir_path, file_name)

def plot_heatmaps(exp_df, ref_df, metrics, datasets, metric_cmaps, results_path):
    # For each dataset
    for dataset in datasets:
        # Filter exp_df by dataset and remove rows where n_neighbors is NaN
        df_dataset = exp_df[(exp_df['dataset_name'] == dataset) & exp_df['n_neighbors'].notna()]

        # Calculate the grid size
        n_metrics = len(metrics)
        ncols = len(df_dataset['dr_name'].unique())  # Number of columns is equal to the number of dr_names
        nrows = n_metrics  # Number of rows is equal to the number of metrics

        # Initialize the figure
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*4), sharey=True)  # Increased figsize

        # For each metric
        for i, metric in enumerate(metrics):
            ref_value_raw = ref_df[ref_df['dataset_name'] == dataset][metric].values[0]  # Get raw reference value
            ref_value = format_value(ref_value_raw)  # Format the reference value
            # For each dr_name (technique)
            for j, dr_name in enumerate(df_dataset['dr_name'].unique()):
                # Select rows where dr_name matches
                df_technique = df_dataset[df_dataset['dr_name'] == dr_name]

                # Pivot to create a matrix
                metric_matrix = df_technique.pivot_table(index='n_components', columns='n_neighbors', values=metric, aggfunc='mean')
                
                # Apply the format_value function to the values in the matrix
                metric_matrix_fmt = metric_matrix.applymap(format_value)

                # Plot the heatmap
                ax = axes[i, j]  # [row_index, col_index] - i.e., [metric, technique]
                sns.heatmap(metric_matrix, annot=metric_matrix_fmt, fmt="s", cmap=metric_cmaps[metric], ax=ax)
                ax.set_title(f"{dr_name} - {metric}\nReference: {ref_value}")  # Include reference value in the title
                ax.set_xlabel('n_neighbors')

                # Set x-ticks and y-ticks as integers, in the middle of each box
                xtick_positions = [x + 0.5 for x in range(len(metric_matrix.columns))]
                ytick_positions = [y + 0.5 for y in range(len(metric_matrix.index))]
                ax.set_xticks(xtick_positions)
                ax.set_xticklabels(metric_matrix.columns.astype(int))
                ax.set_yticks(ytick_positions)
                ax.set_yticklabels(metric_matrix.index.astype(int))

        # Save the figure to a file after iterating all metrics
        plt.tight_layout()

        dir_path = os.path.join(results_path, 'heatmaps')
        file_name = f"{dataset}_heatmap.png"

        # Save the plot using the helper function
        save_plot(fig, dir_path, file_name)

def plot_aggregated_lineplot(exp_df, metric, results_path):
    sns.set(style="whitegrid")
    sns.set_palette(sns.color_palette("husl", 8))

    # Calculate the percentage of retained components over features
    exp_df['n_components_percentage'] = exp_df['n_components'] / exp_df["n_instances"] * 100

    # Determine the range for binning
    end = max(exp_df['n_components_percentage'])
    
    # Create bins for n_components_percentage and calculate the average within each bin
    bin_step = 5 # Regular interval for bins
    bins = np.arange(0, end + bin_step, bin_step)
    exp_df['n_components_percentage_bin'] = pd.cut(exp_df['n_components_percentage'], bins=bins)
    exp_df['n_components_percentage_avg'] = exp_df.groupby('n_components_percentage_bin')['n_components_percentage'].transform('mean').round(2)

    plt.figure(figsize=(10, 6))
        
    # Group the data by dimensionality reduction technique and average percentage
    grouped = exp_df.groupby(['dr_name', 'n_components_percentage_avg'], as_index=False).mean()
        
    # Plot the data
    sns.lineplot(data=grouped, x='n_components_percentage_avg', y=metric, hue='dr_name')
    plt.title(f'Line plot of {metric} aggregated over datasets')
    plt.xlabel('n_components (%)')
    plt.ylabel(metric.capitalize())
    plt.xticks(bins) # Set regular x-ticks
    plt.legend(title='Techniques')

    # Create the directory path and save the plot
    dir_path = os.path.join(results_path, 'aggregated_line_plots')
    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(os.path.join(dir_path, f'aggregated_{metric}_lineplot.png'))
    plt.close()

def plot_combined_heatmap(exp_df, metric, results_path):

    # Calculate the percentage of retained components
    exp_df['n_components_percentage'] = exp_df['n_components'] / exp_df["n_instances"] * 100
    exp_df['n_neighbors_percentage'] = exp_df['n_neighbors'] / exp_df["n_instances"] * 100

    # Determine the range for binning
    end_components = max(exp_df['n_components_percentage'])
    bins_components = np.arange(0, end_components * 1.05, end_components / 20)
    exp_df['n_components_percentage_bin'] = pd.cut(exp_df['n_components_percentage'], bins=bins_components)

    end_neighbors = max(exp_df['n_neighbors_percentage'])
    bins_neighbors = np.arange(0, end_neighbors * 1.05, end_neighbors / 20)
    exp_df['n_neighbors_percentage_bin'] = pd.cut(exp_df['n_neighbors_percentage'], bins=bins_neighbors)

    # Filter rows where n_neighbors is not NaN
    df_filtered = exp_df[exp_df['n_neighbors'].notna()]

    # Convert 'n_components' to percentage
    df_filtered['n_components_percentage'] = df_filtered['n_components'] / df_filtered['n_instances'] * 100

    # Determine the unique DR techniques (dr_name)
    dr_names = df_filtered['dr_name'].unique()

    # Initialize the figure with 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharey=True, sharex=True)
    axes = axes.flatten()

    # For each dr_name (technique)
    for j, dr_name in enumerate(dr_names):

        # Select rows where dr_name matches
        df_technique = df_filtered[df_filtered['dr_name'] == dr_name].copy()
        
        # Pivot to create a matrix, aggregate across datasets
        metric_matrix = df_technique.pivot_table(index='n_components_percentage_bin', columns='n_neighbors_percentage_bin', values=metric, aggfunc='mean')

        # Plot the heatmap
        ax = axes[j]
        sns.heatmap(metric_matrix, annot=True, fmt=".1f", cmap='YlGnBu', ax=ax)
        ax.set_title(f"{dr_name} - {metric}")
        ax.set_xlabel('n_neighbors (%)')
        ax.set_ylabel('n_components (%)')

        # Set x-ticks and y-ticks as integers, at the edge of each box
        ax.set_xticks(range(len(metric_matrix.columns)))
        ax.set_xticklabels([str(x.left) for x in metric_matrix.columns])
        ax.set_yticks(range(len(metric_matrix.index)))
        ax.set_yticklabels([str(y.left) for y in metric_matrix.index])

    # Save the figure
    plt.tight_layout()
    dir_path = os.path.join(results_path, 'combined_heatmaps')
    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(os.path.join(dir_path, f'combined_{metric}_heatmap.png'))
    plt.close()

def plot_aggregated_lineplot(exp_df, metric, results_path):

    sns.set(style="whitegrid")
    sns.set_palette(sns.color_palette("husl", 8)) # Set the color palette 

    # Calculate the percentage of retained components
    exp_df['n_components_percentage'] = exp_df['n_components'] / exp_df["n_instances"] * 100 

    # Determine the range for binning
    end = max(exp_df['n_components_percentage'])   

    # Create bins for n_components_percentage and calculate the average within each bin, rounding to two decimal places

    # Now we create 101 edges to form 100 intervals
    exp_df['n_components_percentage_bin'] = pd.cut(exp_df['n_components_percentage'], bins=np.linspace(0, end, 101))
    exp_df['n_components_percentage_avg'] = exp_df.groupby('n_components_percentage_bin')['n_components_percentage'].transform('mean').round(2) 

    plt.figure(figsize=(10, 6))       

    # Group the data by dimensionality reduction technique and average percentage
    grouped = exp_df.groupby(['dr_name', 'n_components_percentage_avg'], as_index=False).mean()       

    # Plot the data
    sns.lineplot(data=grouped, x='n_components_percentage_avg', y=metric, hue='dr_name')
    # plt.title(f'Line plot of {metric} aggregated over datasets')
    plt.xlabel('n_components (%)')
    plt.ylabel(metric.capitalize())
    plt.legend(title='Techniques') 

    # Set x-ticks for finer granularity, creating 15 intervals
    plt.xticks(np.linspace(0, end, 16)) 

    # Create the directory path and save the plot
    dir_path = os.path.join(results_path, 'aggregated_line_plots')
    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(os.path.join(dir_path, f'aggregated_{metric}_lineplot.png'))
    plt.close()

def plot_aggregated_heatmap(exp_df, metric, results_path):

    # Calculate the percentage of retained components
    exp_df['n_components_percentage'] = exp_df['n_components'] / exp_df["n_instances"] * 100 

    # Determine the range for binning
    end = max(exp_df['n_components_percentage'])

    # Create bins for n_components_percentage with regular intervals
    # We'll create 11 edges to form 10 intervals
    bins = np.linspace(0, end, 11)
    exp_df['n_components_percentage_bin_idx'] = pd.cut(exp_df['n_components_percentage'], bins=bins, labels=False) 

    # Create a pivot table to rearrange data for the heatmap
    pivot_table = exp_df.pivot_table(index='dr_name', columns='n_components_percentage_bin_idx', values=metric, aggfunc='mean')

    # Ensure all 10 columns are present
    pivot_table = pivot_table.reindex(columns=range(10))

    # Create another pivot table with formatted values for annotation
    pivot_table_fmt = pivot_table.applymap(format_value)

    plt.figure(figsize=(10, 6)) 

    # Plot the heatmap using the numeric pivot_table, and annotate with pivot_table_fmt
    sns.heatmap(pivot_table, annot=pivot_table_fmt, fmt="s", cmap='YlGnBu', cbar_kws={'label': metric.capitalize()})

    # Set x-tick labels to represent the range for each bin
    plt.xticks(np.arange(len(bins)), [round(b, 2) for b in bins]) 
    plt.xlabel('n_components (%)')
    plt.ylabel('Techniques')

    # Create the directory path and save the plot
    dir_path = os.path.join(results_path, 'aggregated_heatmaps')
    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(os.path.join(dir_path, f'aggregated_{metric}_heatmap.png'))
    plt.close()

def plot_aggregated_boxplot(exp_df, metric, results_path):
    sns.set(style="whitegrid")
    sns.set_palette(sns.color_palette("husl", 8)) # Set the color palette

    plt.figure(figsize=(12, 8))

    sns.boxplot(x='dr_name', y=metric, data=exp_df)
    plt.title(f'Box Plot of {metric} by DR Techniques')
    plt.xlabel('DR Techniques')
    plt.ylabel(metric.capitalize())
    plt.grid(True)  # Show grid lines

    dir_path = os.path.join(results_path, 'aggregated_boxplots')
    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(os.path.join(dir_path, f'{metric}_boxplot.png'))
    plt.close()
    
def plot_boxplot_ref(exp_df, ref_df, metric, results_path):
    sns.set(style="whitegrid")
    sns.set_palette(sns.color_palette("husl", 8))  # Set the color palette

    plt.figure(figsize=(12, 8))

    # Compute the improvement
    # Merge the reference values with the experimental data on the dataset_name
    exp_df = pd.merge(exp_df, ref_df[['dataset_name', metric]], on='dataset_name', suffixes=('', '_ref'))

    # Calculate the improvement with respect to the reference value in percentage
    # Determine the operation based on the metric
    if metric == 'RMSE':
        # Lower RMSE is better
        exp_df[metric] = ((exp_df[metric + '_ref'] - exp_df[metric]) / exp_df[metric + '_ref']) * 100
    else:
        # Higher metric values (like F1) are better
        exp_df[metric] = ((exp_df[metric] - exp_df[metric + '_ref']) / exp_df[metric + '_ref']) * 100

    sns.boxplot(x='dr_name', y=metric, data=exp_df)
    # plt.title(f'Box Plot of {metric} Improvement by DR Techniques')
    plt.xlabel('DR Techniques')
    plt.ylabel(f'{metric.capitalize()} Improvement (%)')

    # Get the current y-axis tick locations
    y_ticks = plt.yticks()[0]
    # Compute the new tick locations by halving the existing interval
    new_y_ticks = np.arange(y_ticks[0], y_ticks[-1], (y_ticks[1] - y_ticks[0]) / 2)
    plt.yticks(new_y_ticks)
    plt.grid(True)  # Show grid lines

    dir_path = os.path.join(results_path, 'aggregated_boxplots')
    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(os.path.join(dir_path, f'{metric}_improvement_boxplot.png'))
    plt.close()

def plot_bubble(exp_df, ref_df, metric, results_path):
    sns.set(style="whitegrid")
    sns.set_palette(sns.color_palette("husl"))

    plt.figure(figsize=(10, 6))

    # Compute the improvement
    # Merge the reference values with the experimental data on the dataset_name
    exp_df = pd.merge(exp_df, ref_df[['dataset_name', metric]], on='dataset_name', suffixes=('', '_ref'))

    # Calculate the improvement with respect to the reference value in percentage
    # Determine the operation based on the metric
    if metric == 'RMSE':
        # Lower RMSE is better
        exp_df[metric] = ((exp_df[metric + '_ref'] - exp_df[metric]) / exp_df[metric + '_ref']) * 100
    else:
        # Higher metric values (like F1) are better
        exp_df[metric] = ((exp_df[metric] - exp_df[metric + '_ref']) / exp_df[metric + '_ref']) * 100

    # Plot the scatter plot without considering the size attribute
    plot = sns.scatterplot(data=exp_df, x='dr_name', y=metric, hue='dataset_name', legend='brief')

    # plt.title(f'Bubble plot of {metric} improvement across different DR techniques')
    plt.xlabel('DR Techniques')
    plt.ylabel(f'{metric.capitalize()} Improvement (%)')

    plt.legend(title='Datasets') # Let Seaborn handle the legend creation

    # Get the current y-axis tick locations
    y_ticks = plt.yticks()[0]
    # Compute the new tick locations by halving the existing interval
    new_y_ticks = np.arange(y_ticks[0], y_ticks[-1], (y_ticks[1] - y_ticks[0]) / 2)
    plt.yticks(new_y_ticks)

    # Add a horizontal grid for better visualization
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()

    save_path = os.path.join(results_path, 'bubble_plots')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{metric}_bubble.png'))
    plt.close()

def plot_bubble_category(exp_df, ref_df, metric, dataset_categories, results_path):
    sns.set(style="whitegrid")
    sns.set_palette(sns.color_palette("husl"))

    # Define the property categories
    properties = ['Dimensionality', 'Redundancy', 'Sparsity', 'Autocorrelation']

    # Iterate through the property categories
    for prop in properties:
        # Create a figure with 2 rows and 1 column
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))

        # Define the prefixes for high and low levels
        levels = ['High', 'Low']

        # Variables to keep track of global min and max y-values
        global_min_y = float('inf')
        global_max_y = float('-inf')

        # Temporary axes to find global min and max
        temp_axes = []

        # Iterate through high and low levels
        for idx, level in enumerate(levels):
            ax = axes[idx]

            # Get the dataset names for this property and level
            datasets = dataset_categories[level + ' ' + prop]

            # Filter the experimental data to include only the selected datasets
            filtered_exp_df = exp_df[exp_df['dataset_name'].isin(datasets)]

            # Compute the improvement
            # Merge the reference values with the experimental data on the dataset_name
            filtered_exp_df = pd.merge(filtered_exp_df, ref_df[['dataset_name', metric]], on='dataset_name', suffixes=('', '_ref'))

            # Calculate the improvement with respect to the reference value in percentage
            # Determine the operation based on the metric
            if metric == 'RMSE':
                # Lower RMSE is better
                filtered_exp_df[metric] = ((filtered_exp_df[metric + '_ref'] - filtered_exp_df[metric]) / filtered_exp_df[metric + '_ref']) * 100
            else:
                # Higher metric values (like F1) are better
                filtered_exp_df[metric] = ((filtered_exp_df[metric] - filtered_exp_df[metric + '_ref']) / filtered_exp_df[metric + '_ref']) * 100


            # Plot the scatter plot
            sns.scatterplot(data=filtered_exp_df, x='dr_name', y=metric, hue='dataset_name', ax=ax, legend='brief')

            ax.set_title(f'{level} {prop} - Bubble plot of {metric} improvement')
            ax.set_xlabel('DR Techniques')
            ax.set_ylabel(f'{metric.capitalize()} Improvement (%)')

            # Track the current axes and find the local min and max y-values
            temp_axes.append(ax)
            local_min_y, local_max_y = ax.get_ylim()
            global_min_y = min(global_min_y, local_min_y)
            global_max_y = max(global_max_y, local_max_y)

        # Set the global y-limits for both axes
        for ax in temp_axes:
            ax.set_ylim(global_min_y, global_max_y)
            y_ticks = ax.get_yticks()
            new_y_ticks = np.arange(y_ticks[0], y_ticks[-1], (y_ticks[1] - y_ticks[0]) / 2)
            ax.set_yticks(new_y_ticks)
            ax.grid(axis='y', linestyle='--', alpha=0.6)

        plt.tight_layout()

        save_path = os.path.join(results_path, 'bubble_plots', prop)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{prop}_bubble.png'))

        plt.close()

def plot_box_category(exp_df, ref_df, metric, dataset_categories, results_path):
    sns.set(style="whitegrid")
    sns.set_palette(sns.color_palette("husl"))

    # Define the property categories
    properties = ['Dimensionality', 'Redundancy', 'Sparsity', 'Autocorrelation']

    # Iterate through the property categories
    for prop in properties:
        # Create a figure with 2 rows and 1 column
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))

        # Define the prefixes for high and low levels
        levels = ['High', 'Low']

        # Variables to keep track of global min and max y-values
        global_min_y = float('inf')
        global_max_y = float('-inf')

        # Temporary axes to find global min and max
        temp_axes = []

        # Iterate through high and low levels
        for idx, level in enumerate(levels):
            ax = axes[idx]

            # Get the dataset names for this property and level
            datasets = dataset_categories[level + ' ' + prop]

            # Filter the experimental data to include only the selected datasets
            filtered_exp_df = exp_df[exp_df['dataset_name'].isin(datasets)]

            # Compute the improvement
            # Merge the reference values with the experimental data on the dataset_name
            filtered_exp_df = pd.merge(filtered_exp_df, ref_df[['dataset_name', metric]], on='dataset_name', suffixes=('', '_ref'))

            # Determine the operation based on the metric
            if metric == 'RMSE':
                # Lower RMSE is better
                filtered_exp_df[metric] = ((filtered_exp_df[metric + '_ref'] - filtered_exp_df[metric]) / filtered_exp_df[metric + '_ref']) * 100
            else:
                # Higher metric values (like F1) are better
                filtered_exp_df[metric] = ((filtered_exp_df[metric] - filtered_exp_df[metric + '_ref']) / filtered_exp_df[metric + '_ref']) * 100


            # Plot the scatter plot
            sns.boxplot(data=filtered_exp_df, x='dr_name', y=metric, hue='dr_name', ax=ax)

            ax.set_title(f'{level} {prop} - Box plot of {metric} improvement')
            ax.set_xlabel('DR Techniques')
            ax.set_ylabel(f'{metric.capitalize()} Improvement (%)')

            # Track the current axes and find the local min and max y-values
            temp_axes.append(ax)
            local_min_y, local_max_y = ax.get_ylim()
            global_min_y = min(global_min_y, local_min_y)
            global_max_y = max(global_max_y, local_max_y)

        # Set the global y-limits for both axes
        for ax in temp_axes:
            ax.set_ylim(global_min_y, global_max_y)
            y_ticks = ax.get_yticks()
            new_y_ticks = np.arange(y_ticks[0], y_ticks[-1], (y_ticks[1] - y_ticks[0]) / 2)
            ax.set_yticks(new_y_ticks)
            ax.grid(axis='y', linestyle='--', alpha=0.6)

        plt.tight_layout()

        save_path = os.path.join(results_path, 'box_plots', prop)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{prop}_property_boxplot.png'))

        plt.close()

def plot_line_category(exp_df, metric, dataset_categories, results_path):
    sns.set(style="whitegrid")
    sns.set_palette(sns.color_palette("husl"))
    properties = ['Dimensionality', 'Redundancy', 'Sparsity', 'Autocorrelation']

    for prop in properties:
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))
        levels = ['High', 'Low']
        global_min_y = float('inf')
        global_max_y = float('-inf')
        temp_axes = []

        for idx, level in enumerate(levels):
            ax = axes[idx]
            datasets = dataset_categories[level + ' ' + prop]
            filtered_exp_df = exp_df[exp_df['dataset_name'].isin(datasets)][['dr_name', 'n_components', metric]]
            grouped_exp_df = filtered_exp_df.groupby(['dr_name', 'n_components']).mean().reset_index()
            dr_names = grouped_exp_df['dr_name'].unique()

            # Plotting with matplotlib
            for dr_name in dr_names:
                subset = grouped_exp_df[grouped_exp_df['dr_name'] == dr_name]
                ax.plot(subset['n_components'].values, subset[metric].values, label=dr_name)


            ax.set_title(f'{level} {prop} - Line plot of {metric}')
            ax.set_xlabel('Number of Components')
            ax.set_ylabel(f'{metric.capitalize()}')
            ax.legend()  # Added legend

            temp_axes.append(ax)
            local_min_y, local_max_y = ax.get_ylim()
            global_min_y = min(global_min_y, local_min_y)
            global_max_y = max(global_max_y, local_max_y)

        for ax in temp_axes:
            ax.set_ylim(global_min_y, global_max_y)

        plt.tight_layout()

        save_path = os.path.join(results_path, 'line_plots', prop)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{prop}_line.png'))

        plt.close()

def plot_box_by_n_components_per_technique(exp_df, ref_df, metric, results_path, techniques):
    sns.set(style="whitegrid")
    sns.set_palette(sns.color_palette("husl"))

    # Filter by the selected techniques
    exp_df = exp_df[exp_df['dr_name'].isin(techniques)]

    # Convert the 'n_components' column to integer
    exp_df['n_components'] = exp_df['n_components'].astype(int)

    # Assuming 'n_components' is the correct column name; please adjust if needed
    common_n_components = exp_df['n_components'].drop_duplicates().tolist()

    # Filter the experimental data to include only the selected n_components
    filtered_exp_df = exp_df[exp_df['n_components'].isin(common_n_components)]

    # Compute the improvement
    filtered_exp_df = pd.merge(filtered_exp_df, ref_df[['dataset_name', metric]], on='dataset_name', suffixes=('', '_ref'))
    
    # Determine the operation based on the metric
    if metric == 'RMSE':
        # Lower RMSE is better
        filtered_exp_df[metric] = ((filtered_exp_df[metric + '_ref'] - filtered_exp_df[metric]) / filtered_exp_df[metric + '_ref']) * 100
    else:
        # Higher metric values (like F1) are better
        filtered_exp_df[metric] = ((filtered_exp_df[metric] - filtered_exp_df[metric + '_ref']) / filtered_exp_df[metric + '_ref']) * 100

    # Create a figure
    plt.figure(figsize=(12, 8))

    # Plot the box plot grouped by n_components
    sns.boxplot(data=filtered_exp_df, x='dr_name', y=metric, hue='n_components')

    # plt.title('Box plot of metric improvement grouped by n_components per technique')
    plt.xlabel('DR Techniques')
    plt.ylabel(f'{metric.capitalize()} Improvement (%)')

    plt.tight_layout()

    save_path = os.path.join(results_path, 'box_plots_by_n_components')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, '_n_components_boxplot(have neighbors).png'))

    plt.close()

def rank_techniques(exp_df, metric, results_path):
    # Pivot the DataFrame to have techniques as columns, datasets as rows, and metric values as cells
    pivot_table = exp_df.pivot_table(index='dataset_name', columns='dr_name', values=metric, aggfunc='mean')

    # Determine the ranking order (ascending or descending) based on the metric
    ascending = False if metric == 'f1_score' else True

    # Rank the techniques for each dataset; use the 'rank' method to handle tied ranks appropriately
    ranked_table = pivot_table.rank(axis=1, ascending=ascending)

    # Handle NaN values if there are any
    ranked_table = ranked_table.fillna(0) # You may need to modify this depending on how you want to handle NaN values

    # Convert the ranks to integers (only the values)
    ranked_table = ranked_table.astype(int)

    # Save the ranked table to a CSV file
    save_path = os.path.join(results_path, f'{metric}_ranked_table.csv')
    ranked_table.to_csv(save_path, index_label='dataset_name')

    return ranked_table

def count_improvements(exp_df, ref_df, metric, results_path):
    # List of datasets
    datasets = exp_df['dataset_name'].unique()

    # List of techniques
    techniques = exp_df['dr_name'].unique()

    # Create a DataFrame to store the count of improvements for each technique across datasets
    improvements_df = pd.DataFrame(index=datasets, columns=techniques)
    improvements_df.fillna(0, inplace=True)

    # Determine the comparison operator based on the metric
    compare_op = lambda x, y: x < y if metric == 'RMSE' else x > y

    # Iterate through the datasets
    for dataset in datasets:
        # Filter the rows corresponding to the current dataset
        dataset_df = exp_df[exp_df['dataset_name'] == dataset]
        
        # Get the reference value for the current dataset
        ref_value = ref_df[ref_df['dataset_name'] == dataset][metric]
        if not ref_value.empty:
            ref_value = ref_value.iloc[0]

            # Iterate through the techniques
            for technique in techniques:
                # Get the values for the current technique
                technique_values = dataset_df[dataset_df['dr_name'] == technique][metric]
                # Count the number of results that exceeded or were below the ref value for the current technique
                count = compare_op(technique_values, ref_value).sum()
                # Save the count to the improvements DataFrame
                improvements_df.loc[dataset, technique] = count

    # Optionally, you can save the improvements DataFrame to a file
    improvements_df.to_csv(results_path + '/improvements.csv')

    return improvements_df

