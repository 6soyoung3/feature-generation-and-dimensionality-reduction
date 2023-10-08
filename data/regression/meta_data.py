import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy, iqr, pearsonr

##########################################################################################

'''
Compute and Generate meta_data for statistical properties of datasets for regressions
'''

##########################################################################################


def compute_stats(df):
    # Compute the number of instances and features in the dataset
    n_instances, n_features = df.shape

    # Separate numeric features
    numeric_features = df.select_dtypes(include=np.number)

    # Compute the number and percentage of numeric features
    n_numeric_features = numeric_features.shape[1]
    perc_numeric_features = n_numeric_features / n_features

    X = df.iloc[:, :-1]

    # Calculate redundancy
    corr_matrix = X.corr()
    # Get the total number of correlations
    # Subtract the number of variables (df.shape[1]) as the diagonal of the correlation matrix is always 1
    total_correlations = np.prod(corr_matrix.shape) - n_features
    # Count the number of correlations greater than 0.7 or less than -0.7
    # Anything above abs(0.7) are counted as highly correlated
    high_correlations = np.count_nonzero(
        np.abs(corr_matrix) > 0.7) - n_features
    redundancy = high_correlations / total_correlations

    # Calculate sparsity
    sparsity = 1.0 - (np.count_nonzero(X) / float(X.size))

    # Create a dictionary to store the computed statistics
    stats = {
        'dimensionality': n_features / n_instances,
        'number_of_features': n_features,
        'number_of_instances': n_instances,
        'number_of_numeric_features': n_numeric_features,
        'percentage_of_numeric_features': perc_numeric_features * 100,
        'redundancy': redundancy,
        'sparsity': sparsity
    }

    # Identify constant columns in numeric_features generated when using encoders
    constant_cols = numeric_features.columns[numeric_features.std() == 0]

    # Drop these columns from numeric_features
    numeric_features = numeric_features.drop(constant_cols, axis=1)

    # Compute the average autocorrelation between numeric features and the target variable
    # Autocorrelation measures the linear relationship or similarity between a variable and its lagged values

    autocorrelation = np.mean([pearsonr(
        numeric_features[col], df.iloc[:, -1])[0] for col in numeric_features.columns])
    stats['autocorrelation'] = autocorrelation

    # Regression specific metrics - considering the target variable as the last column
    y = df.iloc[:, -1]
    stats.update({
        'target_mean': y.mean(),
        'target_median': y.median(),
        'target_variance': y.var(),
        'target_iqr': iqr(y),
        'target_skewness': skew(y),
        'target_kurtosis': kurtosis(y),
    })

    # Compute entropy-related statistics for numeric attributes
    # These statistics provide insights into the entropy distribution among the numeric attributes
    # after binning them into intervals and they can help understanding the variability or diversity
    # in the information content of the attributes, indicating how much information or uncertainty is present in each attribute
    attr_entropies = [entropy(pd.cut(df[col], bins=10).value_counts(
    ), base=2) for col in numeric_features.columns]
    stats.update({
        'max_attribute_entropy': np.max(attr_entropies),
        'mean_attribute_entropy': np.mean(attr_entropies),
        'min_attribute_entropy': np.min(attr_entropies),
        'quartile_1_attribute_entropy': np.percentile(attr_entropies, 25),
        'quartile_2_attribute_entropy': np.percentile(attr_entropies, 50),
        'quartile_3_attribute_entropy': np.percentile(attr_entropies, 75),
    })

    # Compute kurtosis-related statistics for numeric attributes
    # Provide insights into the shape and distribution of values in the numeric attributes
    # Higher kurtosis values indicate heavier tails - more outliers or extreme values - compared to a normal distribution
    # Lower kurtosis values indicate lighter tails - fewer outliers or extreme values
    # Help assess the variability and distribution characteristics of the numeric attributes in the dataset
    kurtosis_vals = [kurtosis(numeric_features[col])
                     for col in numeric_features.columns]
    stats.update({
        'max_kurtosis_of_numeric_atts': np.max(kurtosis_vals),
        'mean_kurtosis_of_numeric_atts': np.mean(kurtosis_vals),
        'min_kurtosis_of_numeric_atts': np.min(kurtosis_vals),
        'quartile_1_kurtosis_of_numeric_atts': np.percentile(kurtosis_vals, 25),
        'quartile_2_kurtosis_of_numeric_atts': np.percentile(kurtosis_vals, 50),
        'quartile_3_kurtosis_of_numeric_atts': np.percentile(kurtosis_vals, 75),
    })

    # Compute skewness-related statistics for numeric attributes
    # Provide insights into the symmetry or lack thereof in the distribution of values in the numeric attributes
    # Skewness measures the extent to which a distribution deviates from a symmetric or normal distribution
    # Positive skewness values indicate a right-skewed distribution with a longer right tail
    # Negative skewness values indicate a left-skewed distribution with a longer left tail
    # Help assess the distribution characteristics and skewness of the numeric attributes in the data
    skewness_vals = [skew(numeric_features[col])
                     for col in numeric_features.columns]
    stats.update({
        'max_skewness_of_numeric_atts': np.max(skewness_vals),
        'mean_skewness_of_numeric_atts': np.mean(skewness_vals),
        'min_skewness_of_numeric_atts': np.min(skewness_vals),
        'quartile_1_skewness_of_numeric_atts': np.percentile(skewness_vals, 25),
        'quartile_2_skewness_of_numeric_atts': np.percentile(skewness_vals, 50),
        'quartile_3_skewness_of_numeric_atts': np.percentile(skewness_vals, 75),
    })

    # Compute standard deviation-related statistics for numeric attributes
    # Provide insights into the variability or spread of values in the numeric attributes
    # Standard deviation measures how much the values deviate from the mean
    # Higher standard deviation values indicate greater variability or dispersion
    # Lower standard deviation values indicate less variability or dispersion
    # Help assess the dispersion and spread of the numeric attributes in the dataset
    stddev_vals = [np.std(numeric_features[col])
                   for col in numeric_features.columns]
    stats.update({
        'max_std_dev_of_numeric_atts': np.max(stddev_vals),
        'mean_std_dev_of_numeric_atts': np.mean(stddev_vals),
        'min_std_dev_of_numeric_atts': np.min(stddev_vals),
        'quartile_1_std_dev_of_numeric_atts': np.percentile(stddev_vals, 25),
        'quartile_2_std_dev_of_numeric_atts': np.percentile(stddev_vals, 50),
        'quartile_3_std_dev_of_numeric_atts': np.percentile(stddev_vals, 75),
    })

    # Compute mean-related statistics for numeric attributes
    # Provide insights into the average value or central tendency of the numeric attributes
    # The mean represents the typical value or centre of the distribution for each attribute
    # Help assess the central tendency and average values of the numeric attributes in the dataset
    means_vals = [np.mean(numeric_features[col])
                  for col in numeric_features.columns]
    stats.update({
        'max_means_of_numeric_atts': np.max(means_vals),
        'mean_means_of_numeric_atts': np.mean(means_vals),
        'min_means_of_numeric_atts': np.min(means_vals),
        'quartile_1_means_of_numeric_atts': np.percentile(means_vals, 25),
        'quartile_2_means_of_numeric_atts': np.percentile(means_vals, 50),
        'quartile_3_means_of_numeric_atts': np.percentile(means_vals, 75),
    })

    return stats
