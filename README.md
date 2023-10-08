# feature_generator
Soyoung Park, KCL 2023

## Project Title
https://www.overleaf.com/project/649300dda9aebff1c14077eb

## Research Question
> How does the performance of various Dimensionality Reduction (DR) techniques differ across different types of datasets (classification, regression, time series)?
>
> How do different dimensionality reduction Python libraries compare in terms of their effectiveness and efficiency in preserving essential information and reducing dimensionality across various types of datasets?

1. Comparative Performance: The question focuses on comparing the performance of different DR Python libraries. Having the investigation centred around dimensionality reduction techniques, such as evaluation and comparison of the performance and the effectiveness of multiple libraries includes:
- Preserved Data Integrity
  - the prediction accuracy and precision before and after each DR technique
  - the reduced dataset must still accurately represents the original data and ideally perform better than the original data
- Time Consumption
  - the training and any processing time should be reduced to within a reasonable time

2. Different Types of Datasets: The question suggests that this research will be working with various types of datasets. This allows for a diverse range of data characteristics, such as size and structure, to be considered:
- In Classification:
- In Regression:

## Project Description
This project aims to conduct a comprehensive investigation into DR techniques and their effects on various downstream tasks, such as regression or classification. The primary objective is to replicate and expand upon existing studies in this field, encompassing a broader range of datasets and dimensionality reduction methods. This project focuses on comparing the effectiveness of different DR techniques in terms of their impact on the performance of different types of data analysis tasks

<!-- Furthermore, the project will delve into the relationship between the statistical properties of datasets and the performance of different dimensionality reduction methods. By analyzing this correlation, valuable insights can be gained, potentially leading to the identification of optimal methods for specific dataset characteristics. -->

<!-- One of the key anticipated outcomes of this project is the development of a unified dimensionality reduction suite. This suite will consolidate a wide array of existing methods into a single library, offering a cohesive and standardized interface. This consolidated suite will provide researchers and practitioners with a convenient tool to explore and utilize various dimensionality reduction techniques efficiently. -->

To achieve these goals, the project will involve replicating existing studies, expanding the scope of datasets and methods examined, and analyzing the impact of dimensionality reduction on different downstream tasks. By conducting rigorous experiments and drawing conclusions from the results, the project will provide recommendations on the optimal method(s) to employ based on the statistical properties of a given dataset.

<!-- Overall, this project aims to contribute to the field of dimensionality reduction by deepening our understanding of the effects of different techniques, facilitating the development of a unified library, and offering practical guidelines for selecting suitable methods based on dataset characteristics. -->

### DR Methods
- **Linear**
  - PCA
  - SVD
  - ICA
  - LDA
- **Nonlinear**
  - KPCA
  - MDS
  - Isomap
  - LLE
  - UMAP
 and more...

 ### Evaluation Methodologies
 To evaluate the performance of the DR techniques, different Machine Learning (ML) models trained by different datasets, i.e. the reduced datasets and the original datasets, but with the same learner should be compared to each other.
 
 This leads to assessing the performance of the ML models, including:
- Classification accuracy
- F1 score (precision and recall)
- Error rate
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²) Score
- Mean Absolute Error (MAE)