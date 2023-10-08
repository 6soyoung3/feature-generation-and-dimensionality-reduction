#!/bin/bash

# Specify the python script file location
project_path='/Users/soyoungpark/Desktop/7CCSMPRJ/2023_feature_generator'
python_script="$project_path/experiments/script.py"

# Create the nohup directory if it doesn't exist
nohup_dir="$project_path/experiments/results"
mkdir -p $nohup_dir
echo "Nohup directory: $nohup_dir" # Check the path to nohup directory

# Specify the list of experiments
experiment_list=$(seq 1 10)
start_core=$0

# Specify the list of tasks and models
task_list=("classification" "regression")
model_list=("pca_model" "kpca_model" "ica_model" "svd_model" "lpp_model" "umap_model" "isomap_model" "lle_model" "grp_model")

# Iterate through the list of tasks, models, and experiments
for task in "${task_list[@]}"; do
  for model in "${model_list[@]}"; do
    for experiment_id in $experiment_list; do
        start_core=$((start_core+1))
        echo "Running $model for $task in experiment $experiment_id"
        
        # Run the python script with the task, model, and experiment_id as parameters
        nohup taskset -c $start_core python3 -u $python_script -t "$task" -m "$model" -e "$experiment_id" > "$nohup_dir/${task}_${model}_experiment_$experiment_id.out" 2>&1 &

    done
  done
done
