# Experiments Setup

## 1. Downloading Datasets
- Acquire ten different datasets for classification tasks
    - Acquire ten different datasets designed for classification tasks
- Acquire ten different datasets for regression tasks
    - Acquire ten different datasets designed for regression tasks

## 2. Pre-processing Datasets
- Categorical variables are converted into numerical values
    - Categorical variables, such as those with nominal or ordinal characteristics, are transformed into numerical representations. This conversion allows the data to be processed by machine learning algorithms, as most algorithms work with numerical data.
- Features are standardised
    - Standardisation is a technique that rescales features to have a mean of 0 and a standard deviation of 1
    - For some DR techniques standardisation is required and also promotes fair and consistent comparison among features
    - This ensures that all features are on a similar scale, preventing certain features from dominating the learning process due to their larger magnitude

## 3. Applying Dimensionality Reduction (DR) Techniques
- Each DR technique requires tuning parameters
    - Dimensionality reduction techniques have specific parameters that need to be adjusted for optimal performance
        - These parameters include the number of components (n_components) or the number of neighbours (n_neighbours) to consider during the reduction process
- Group DR techniques based on their parameters
    - To determine which parameters should be considered hyperparameters and tuned, the DR techniques are categorised based on the specific parameters they utilise
        - This grouping aids in identifying the parameters within each technique that require adjustment
- Different DR techniques may require different parameter configurations
    - Since each DR technique operates differently, they may necessitate distinct parameter configurations for optimal results
        - This highlights the importance of tailoring the setup for each parameter type according to the specific requirements of the technique being used
- Each dataset is reduced to multiple forms of reduced datasets
    - Applying the selected DR techniques to each dataset results in multiple variations of the dataset, each representing a different reduction of its original dimensions
        - This allows for a comparative analysis of the impact of dimensionality reduction on subsequent machine learning tasks

## 4. Training Machine Learning (ML) Models
- Different ML algorithms are employed based on the task
    - Machine learning encompasses various algorithms, each suitable for different types of tasks and in this step, the appropriate ML algorithms are selected based on whether the task is classification or regression. 
- Each ML model is trained using different reduced datasets
    - The ML models are trained using the various reduced datasets generated in the previous step
    - Each model is trained independently on different versions of the dataset to observe the impact of dimensionality reduction on the model's performance
    - To prioritise the evaluation of dimensionality reduction (DR) techniques and isolate the effect of the machine learning (ML) algorithm, the tuning of any necessary hyperparameters is omitted and fixed instead

## 5. Evaluating ML Model Performance
- Insights into the relationship between ML algorithms and DR techniques
    - This analysis seeks to gain an understanding of the connection between the choice of ML algorithm and the dimensionality reduction technique applied
    - By evaluating the performance of different ML models trained on the various reduced datasets, insights can be obtained regarding the influence of DR on the effectiveness of different ML algorithms
- Insights into the relationship between DR techniques and datasets
    - Similarly, this examination aims to explore the relationship between the choice of dimensionality reduction technique and the specific dataset being used
    - By comparing the performance of the ML models trained on different reduced datasets derived from the same original dataset, insights can be gained into the impact of various DR techniques on different types of datasets
