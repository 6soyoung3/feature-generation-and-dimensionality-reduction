#!/bin/bash

# Specify the python script file location
project_path='/home/soyoung/Desktop/2023_feature_generator'

python_script="$project_path/experiments/experiment_all.py"
ml_script="$project_path/experiments/ml_reference_script.py"
visualisation_script="$project_path/visualisations/visualisation_script.py"

# Create the nohup directory if it doesn't exist
nohup_dir="$project_path/experiments/results"
mkdir -p $nohup_dir
echo "Nohup directory: $nohup_dir" # Check the path to nohup directory

task=$1

# Specify the list of experiments
experiment_list=$(seq 1 1)

# Specify the data location
data_path='/home/soyoung/Desktop/2023_feature_generator/data'

# Create task subdirectory inside nohup directory
task_nohup_dir="$nohup_dir/${task}"
mkdir -p $task_nohup_dir

# Generate the reference ML results
# python3 "$ml_script" -t "$task" -d "$data_path" > "$task_nohup_dir/ml_script_output.out"

# start_core=0

# Run the experiments
for experiment_id in $experiment_list; do
  # echo "Running for $task in experiment $experiment_id with core $start_core"
  echo "Running for $task in experiment $experiment_id"

  # Create experiment subdirectory inside task directory
  experiment_nohup_dir="$task_nohup_dir/experiment_${experiment_id}"
  mkdir -p $experiment_nohup_dir

  python3 "$python_script" -t "$task" -d "$data_path" -id "$experiment_id" > "$experiment_nohup_dir/experiment_${experiment_id}.out" 2>&1
  # nohup taskset -c $start_core python3 "$python_script" -t "$task" -d "$data_path" -id "$experiment_id" > "$experiment_nohup_dir/experiment_${experiment_id}.out" 2>&1 &
  # start_core=$((start_core + 1))
done

# python3 "$visualisation_script" -t "$task" > "$project_path/visualisations/visualisation_output.out"