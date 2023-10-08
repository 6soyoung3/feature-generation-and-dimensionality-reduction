##########################################################################################
'''
Pre-step for Classification

- Measure the size and the dimensionality of each dataset
- Show the distribution of the current dataset i.e. linearity and non-linearity
- Calculate the redundancy (correlations) and sparsity of each dataset
- Any information of the datasets before applying DR must be displayed here

'''
##########################################################################################

##########################################################################################
'''
Import Libraries
'''
##########################################################################################

import os
import sys
sys.path.append('/home/soyoung/Desktop/2023_feature_generator')

from meta_data import compute_stats
from experiments.utils.dataset_utils import compute_and_save_data_stats

# Changed the relative path to start from the current directory
summaries_dir = "data/classification/summaries"

# Convert summaries_dir to an absolute path
abs_summaries_dir = os.path.abspath(summaries_dir)

# If the directory doesn't exist, create it
if not os.path.exists(abs_summaries_dir):
    os.makedirs(abs_summaries_dir)

##########################################################################################
'''
Generate datasets properties summary table
'''
##########################################################################################

folder_path = "data/classification/datasets"

compute_and_save_data_stats(folder_path, summaries_dir, compute_stats)