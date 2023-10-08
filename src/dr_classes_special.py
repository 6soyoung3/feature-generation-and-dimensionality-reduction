import concurrent.futures

import numpy as np
import pandas as pd

import os

from sklearn.decomposition import PCA as pca, FastICA as ica, KernelPCA as kpca, TruncatedSVD as svd
from sklearn.manifold import Isomap, LocallyLinearEmbedding as lle
from lpproj import LocalityPreservingProjection as lpp
from umap import UMAP as umap

from abc import ABC, abstractmethod

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('/home/soyoung/Desktop/2023_feature_generator')
from experiments.utils.ml_tasks_special import *

class DR(ABC):
    def __init__(self):
        self.models = pd.Series(dtype=object)
        self.requires_neighbors = False
        self.reduced_data = pd.DataFrame(dtype=object)
                
    @abstractmethod
    def create_dr_model(self, n_components, best_params):
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def create_reduced_dataset_name(self, dataset_name, params):
        raise NotImplementedError("Subclasses must implement this method.")

    def reduce(self, dataset_name, X_train, X_test):
        self.reduced_data = pd.DataFrame(
            columns=['dr_name', 'dataset_name', 'n_components', 'n_neighbors', 'reduced_X_train', 'reduced_X_test'])
        
        if X_train.shape[1] > 50: n_components_range = list(range(2, X_train.shape[1]//3, X_train.shape[1]//30))
        else: n_components_range = list(range(2, X_train.shape[1]//2, 2))
            
        neighbors_list = [int(np.sqrt(X_train.shape[0])), int(
            np.log(X_train.shape[0])), int(np.cbrt(X_train.shape[0]))]
        neighbors_list.sort()

        time_exceeded = False

        for n_component in n_components_range:
            params = {"n_components": n_component}

            if self.requires_neighbors:
                for neighbor in neighbors_list:
                    params["n_neighbors"] = neighbor

                    time_exceeded, reduced_X_train = self.fit_transform(
                        dataset_name, X_train, params)
                        
                    if time_exceeded:
                        return self.reduced_data

                    reduced_X_test = self.transform(dataset_name, X_test, params)

                    
                    new_row = pd.DataFrame({
                        'dr_name': self.__class__.__name__,
                        'dataset_name': dataset_name,
                        'n_components': params['n_components'],
                        'n_neighbors': params.get('n_neighbors', None),
                        'reduced_X_train': [reduced_X_train],
                        'reduced_X_test': [reduced_X_test]
                    }, index=[0])
                    

                    self.reduced_data = pd.concat([self.reduced_data, new_row], ignore_index=True)
            else:
                time_exceeded, reduced_X_train = self.fit_transform(
                    dataset_name, X_train, params)

                if time_exceeded:
                    return self.reduced_data
                    
                reduced_X_test = self.transform(dataset_name, X_test, params)

                
                new_row = pd.DataFrame({
                    'dr_name': self.__class__.__name__,
                    'dataset_name': dataset_name,
                    'n_components': params['n_components'],
                    'n_neighbors': params.get('n_neighbors', None),
                    'reduced_X_train': [reduced_X_train],
                    'reduced_X_test': [reduced_X_test]
                }, index=[0])
                

                self.reduced_data = pd.concat([self.reduced_data, new_row], ignore_index=True)

        return self.reduced_data
    
    def experiment(self, task, dataset_name, X_train, X_test, y_train, y_test, file_path):
        self.reduced_data = pd.DataFrame(
            columns=['dr_name', 'dataset_name', 'n_components', 'n_neighbors', 'reduced_X_train', 'reduced_X_test'])
        
        if X_train.shape[1] > 50: n_components_range = list(range(2, X_train.shape[1]//3, X_train.shape[1]//30))
        else: n_components_range = list(range(2, X_train.shape[1]//2, 2))
            
        neighbors_list = [int(np.sqrt(X_train.shape[0])), int(
            np.log(X_train.shape[0])), int(np.cbrt(X_train.shape[0]))]
        neighbors_list.sort()

        time_exceeded = False

        for n_component in n_components_range:
            params = {"n_components": n_component}

            if self.requires_neighbors:
                for neighbor in neighbors_list:
                    params["n_neighbors"] = neighbor

                    time_exceeded, reduced_X_train = self.fit_transform(
                        dataset_name, X_train, params)
                        
                    if time_exceeded:
                        return self.reduced_data

                    reduced_X_test = self.transform(dataset_name, X_test, params)

                    data_info = {
                        'dr_name': self.__class__.__name__,
                        'dataset_name': dataset_name,
                        'n_components': params['n_components'],
                        'n_neighbors': params.get('n_neighbors', None),
                    }

                    if task == "classification": ml_classification(data_info, reduced_X_train, reduced_X_test, y_train, y_test, file_path)
                    else: ml_regression(data_info, reduced_X_train, reduced_X_test, y_train, y_test, file_path)
            else:
                time_exceeded, reduced_X_train = self.fit_transform(
                    dataset_name, X_train, params)

                if time_exceeded:
                    return self.reduced_data
                    
                reduced_X_test = self.transform(dataset_name, X_test, params)

                data_info = {
                        'dr_name': self.__class__.__name__,
                        'dataset_name': dataset_name,
                        'n_components': params['n_components'],
                        'n_neighbors': params.get('n_neighbors', None),
                    }
                
                if task == "classification": ml_classification(data_info, reduced_X_train, reduced_X_test, y_train, y_test, file_path)
                else: ml_regression(data_info, reduced_X_train, reduced_X_test, y_train, y_test, file_path)

    def fit_transform(self, dataset_name, X_train, params):

        dr_model = self.create_dr_model(**params)

        reduced_dataset_name = self.create_reduced_dataset_name(dataset_name, params)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(dr_model.fit_transform, X_train)
            try:
                reduced_X_train = future.result(timeout=180)
            except concurrent.futures.TimeoutError:
                print("** fit_transform took too long **")
                return True, None  # Indicate that timeout exceeded

        self.models[reduced_dataset_name] = dr_model

        print("fit_transform : ", reduced_dataset_name)

        return False, reduced_X_train  # Indicate that process completed within time limit
    
    def transform(self, dataset_name, X_test, params):     
        reduced_dataset_name = self.create_reduced_dataset_name(dataset_name, params)
        if reduced_dataset_name not in self.models: return
        dr_model = self.models[reduced_dataset_name]
        reduced_X_test = self._apply_transform(
            dr_model, X_test, dataset_name, params)

        return reduced_X_test

    def _apply_transform(self, dr_model, X_test, dataset_name, params):

        reduced_X_test = dr_model.transform(X_test)

        reduced_dataset_name = self.create_reduced_dataset_name(dataset_name, params)

        print("transform : ", reduced_dataset_name)

        return reduced_X_test
class KPCA(DR):
    def create_dr_model(self, n_components, **kwargs):
        return kpca(kernel='rbf', n_components=n_components)

    def create_reduced_dataset_name(self, dataset_name, params):
        return dataset_name + "_KPCA_" + str(params['n_components'])


class PCA(DR):
    def create_dr_model(self, n_components, **kwargs):
        return pca(n_components=n_components)

    def create_reduced_dataset_name(self, dataset_name, params):
        return dataset_name + "_PCA_" + str(params['n_components'])


class ICA(DR):
    def create_dr_model(self, n_components, **kwargs):
        # the default of max_iter is 200
        # the default of tol is 1e-4
        return ica(n_components=n_components)

    def create_reduced_dataset_name(self, dataset_name, params):
        return dataset_name + "_ICA_" + str(params['n_components'])


class SVD(DR):
    def create_dr_model(self, n_components, **kwargs):
        return svd(n_components=n_components)

    def create_reduced_dataset_name(self, dataset_name, params):
        return dataset_name + "_SVD_" + str(params['n_components'])


'''
- The fit_transform method of a dimensionality reduction model both learns the parameters
necessary for the transformation and applies the transformation to the same data
- Not all dimensionality reduction models can apply the learned transformation to new data
- Multidimensional Scaling (MDS) doesn't have a transform method because it's a non-parametric method
that doesn't learn a parameter-based mapping from high-dimensional space to low-dimensional space
- Therefore, there's no general transformation rule in MDS that can be applied to new, unseen data
- It's based on a dissimilarity matrix calculated for a specific dataset
- Some other dimensionality reduction techniques, like t-SNE, also do not offer a transform method for the same reasons
'''


class ISOMAP(DR):
    def __init__(self):
        super().__init__()
        self.requires_neighbors = True

    def create_dr_model(self, n_components, n_neighbors, **kwargs):
        return Isomap(n_components=n_components,
                      n_neighbors=n_neighbors)

    def create_reduced_dataset_name(self, dataset_name, params):
        return dataset_name + "_ISOMAP_" + str(params['n_components']) + '_' + str(params['n_neighbors'])


class LLE(DR):
    def __init__(self):
        super().__init__()
        self.requires_neighbors = True

    def create_dr_model(self, n_components, n_neighbors, **kwargs):
        return lle(n_components=n_components,
                   n_neighbors=n_neighbors,
                   )

    def create_reduced_dataset_name(self, dataset_name, params):
        return dataset_name + "_LLE_" + str(params['n_components']) + '_' + str(params['n_neighbors'])


class LPP(DR):
    def __init__(self):
        super().__init__()
        self.requires_neighbors = True

    def create_dr_model(self, n_components, n_neighbors=None, **kwargs):
        return lpp(n_components=n_components,
                   n_neighbors=n_neighbors)

    def create_reduced_dataset_name(self, dataset_name, params):
        return dataset_name + "_LPP_" + str(params['n_components']) + '_' + str(params['n_neighbors'])


class UMAP(DR):
    def __init__(self):
        super().__init__()
        self.requires_neighbors = True

    def create_dr_model(self, n_components, n_neighbors=None, **kwargs):
        return umap(n_components=n_components,
                    n_neighbors=n_neighbors,
                    )

    def create_reduced_dataset_name(self, dataset_name, params):
        return dataset_name + "_UMAP_" + str(params['n_components']) + '_' + str(params['n_neighbors'])
