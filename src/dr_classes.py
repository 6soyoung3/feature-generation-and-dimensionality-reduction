import concurrent.futures

import time
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

class DR(ABC):
    def __init__(self):
        self.models = pd.Series(dtype=object)
        self.requires_neighbors = False
        self.reduced_trains = pd.Series(dtype=object)
        self.reduced_tests = pd.Series(dtype=object)

        self.time_taken_fit_transform = pd.DataFrame(
            columns=['dr_name', 'dataset_name', 'n_components', 'n_neighbors', 'fit_transform_time'])
        self.time_taken_transform = pd.DataFrame(
            columns=['dr_name', 'dataset_name', 'n_components', 'n_neighbors', 'transform_time'])
        
    @abstractmethod
    def create_dr_model(self, n_components, best_params):
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def create_index(self, dataset_name, params):
        raise NotImplementedError("Subclasses must implement this method.")

    def fit_transform(self, X_trains):
        # model_name = self.__class__.__name__
        # dir_path = self.create_path(experiment_id, task_type=task_type)
        print(self.time_taken_fit_transform.columns)

        for dataset_name, X_train in X_trains.items():
            if X_train.shape[1] > 50: n_components_range = list(range(2, X_train.shape[1]//3, X_train.shape[1]//30))
            else: n_components_range = list(range(2, X_train.shape[1]//2, 2))
            
            neighbors_list = [int(np.sqrt(X_train.shape[0])), int(
                np.log(X_train.shape[0])), int(np.cbrt(X_train.shape[0]))]
            neighbors_list.sort()

            if self.requires_neighbors:
                self.neighbors_used[dataset_name] = []

            time_exceeded = False
            for n_component in n_components_range:
                params = {"n_components": n_component}

                if self.requires_neighbors:
                    for neighbor in neighbors_list:
                        params["n_neighbors"] = neighbor
                        time_exceeded = self._fit_transform_model(
                            X_train, dataset_name, params)
                        if not time_exceeded: self.neighbors_used[dataset_name].append(
                            (n_component, neighbor))

                        if time_exceeded:
                            break
                else:
                    time_exceeded = self._fit_transform_model(
                        X_train, dataset_name, params)

                if time_exceeded:
                    break

            if time_exceeded:
                continue  # move on to the next dataset

        # save the reduced dataset as a .csv file
        # os.makedirs(f"{experiments_dir}/train", exist_ok=True)
        # self.reduced_trains.to_csv(f"{experiments_dir}/train/{model_name.lower()}.csv", index=False)

        return self.reduced_trains, self.time_taken_fit_transform

    def _fit_transform_model(self, X_train, dataset_name, params):
        # print("parameters: ", params)

        dr_model = self.create_dr_model(**params)

        index = self.create_index(dataset_name, params)

        start = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(dr_model.fit_transform, X_train)
            try:
                reduced_X_train = future.result(timeout=180)
            except concurrent.futures.TimeoutError:
                print("** fit_transform took too long **")
                return True  # Indicate that timeout exceeded

        elapsed = time.time() - start

        # print("time taken around: ", int(elapsed), "secs")

        self.models[index] = dr_model
        # print("Saved model in index: ", index)
        self.reduced_trains[index] = reduced_X_train

        new_row = pd.DataFrame({
            'dr_name' : self.__class__.__name__,
            'dataset_name': dataset_name,
            'n_components': params['n_components'],
            'n_neighbors': params.get('n_neighbors', None),
            'fit_transform_time': elapsed
        }, index=[0])

        # print(new_row.values)
        row_str = ",".join(str(val) for val in new_row.iloc[0])
        print(row_str)

        self.time_taken_fit_transform = pd.concat(
            [self.time_taken_fit_transform, new_row], ignore_index=True)

        # print("fit_transform : ", index)

        return False  # Indicate that process completed within time limit
    
    def transform(self, X_tests):
        # model_name = self.__class__.__name__
        # dir_path = self.create_path(experiment_id, task_type=task_type)
        print(self.time_taken_transform.columns)
        
        for dataset_name, X_test in X_tests.items():
            if self.requires_neighbors:
                while self.neighbors_used[dataset_name]:
                    n_component, neighbor = self.neighbors_used[dataset_name].pop(
                        0)
                    params = {"n_components": n_component,
                              "n_neighbors": neighbor}
                    index = self.create_index(dataset_name, params)
                    dr_model = self.models[index]
                    self._apply_transform(
                        dr_model, X_test, dataset_name, params)
            else:
                if X_test.shape[1] > 50: n_components_range = list(range(2, X_test.shape[1]//3, X_test.shape[1]//30))
                else: n_components_range = list(range(2, X_test.shape[1]//2, 2))
                for n_component in n_components_range:
                    params = {"n_components": n_component}
                    index = self.create_index(dataset_name, params)
                    if index not in self.models: break
                    dr_model = self.models[index]
                    self._apply_transform(
                        dr_model, X_test, dataset_name, params)

        # save the reduced dataset as a .csv file
        # os.makedirs(f"{experiments_dir}/test", exist_ok=True)
        # self.reduced_tests.to_csv(f"{experiments_dir}/test/{model_name.lower()}.csv", index=False)
        
        return self.reduced_tests, self.time_taken_transform

    def _apply_transform(self, dr_model, X_test, dataset_name, params):

        start = time.time()

        reduced_X_test = dr_model.transform(X_test)

        elapsed = time.time() - start

        index = self.create_index(dataset_name, params)

        self.reduced_tests[index] = reduced_X_test

        new_row = pd.DataFrame({
            'dr_name' : self.__class__.__name__,
            'dataset_name': dataset_name,
            'n_components': params['n_components'],
            'n_neighbors': params.get('n_neighbors', None),
            'transform_time': elapsed
        }, index=[0])

        row_str = ",".join(str(val) for val in new_row.iloc[0])
        print(row_str)

        self.time_taken_transform = pd.concat(
            [self.time_taken_transform, new_row], ignore_index=True)

        # print("transform : ", index)

        # save the reduced dataset as a .csv file
        # self.save_as_csv(dir_path, dataset_name, 'test', index, reduced_X_test)

    def create_path(self, experiment_id, task_type=None):
        assert task_type is not None, "Task type must be specified."
        # project_path = '/home/soyoung/Desktop/2023_feature_generator/'
        project_path = '/Users/soyoungpark/Desktop/7CCSMPRJ/2023_feature_generator/'
        experiments_dir = os.path.join(project_path, "experiments/results", task_type)
        dir_path = os.path.join(experiments_dir, f"experiment_{experiment_id}")
        os.makedirs(dir_path, exist_ok=True)
        return dir_path
    
    # def save_as_csv(self, dir_path, dataset_name, phase, index, data):
    #     file_path = os.path.join(dir_path, dataset_name, phase)
    #     os.makedirs(file_path, exist_ok=True)
    #     file_path = os.path.join(file_path, f"{index}.csv")
    #     pd.DataFrame(data).to_csv(file_path, index=False)


class KPCA(DR):
    def create_dr_model(self, n_components, **kwargs):
        return kpca(kernel='rbf', n_components=n_components)

    def create_index(self, dataset_name, params):
        return dataset_name + "_KPCA_" + str(params['n_components'])


class PCA(DR):
    def create_dr_model(self, n_components, **kwargs):
        return pca(n_components=n_components)

    def create_index(self, dataset_name, params):
        return dataset_name + "_PCA_" + str(params['n_components'])


class ICA(DR):
    def create_dr_model(self, n_components, **kwargs):
        # the default of max_iter is 200
        # the default of tol is 1e-4
        return ica(n_components=n_components)

    def create_index(self, dataset_name, params):
        return dataset_name + "_ICA_" + str(params['n_components'])


class SVD(DR):
    def create_dr_model(self, n_components, **kwargs):
        return svd(n_components=n_components)

    def create_index(self, dataset_name, params):
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
        self.neighbors_used = {}

    def create_dr_model(self, n_components, n_neighbors, **kwargs):
        return Isomap(n_components=n_components,
                      n_neighbors=n_neighbors)

    def create_index(self, dataset_name, params):
        return dataset_name + "_ISOMAP_" + str(params['n_components']) + '_' + str(params['n_neighbors'])


class LLE(DR):
    def __init__(self):
        super().__init__()
        self.requires_neighbors = True
        self.neighbors_used = {}

    def create_dr_model(self, n_components, n_neighbors, **kwargs):
        return lle(n_components=n_components,
                   n_neighbors=n_neighbors,
                   )

    def create_index(self, dataset_name, params):
        return dataset_name + "_LLE_" + str(params['n_components']) + '_' + str(params['n_neighbors'])


class LPP(DR):
    def __init__(self):
        super().__init__()
        self.requires_neighbors = True
        self.neighbors_used = {}

    def create_dr_model(self, n_components, n_neighbors=None, **kwargs):
        return lpp(n_components=n_components,
                   n_neighbors=n_neighbors)

    def create_index(self, dataset_name, params):
        return dataset_name + "_LPP_" + str(params['n_components']) + '_' + str(params['n_neighbors'])


class UMAP(DR):
    def __init__(self):
        super().__init__()
        self.requires_neighbors = True
        self.neighbors_used = {}

    def create_dr_model(self, n_components, n_neighbors=None, **kwargs):
        return umap(n_components=n_components,
                    n_neighbors=n_neighbors,
                    )

    def create_index(self, dataset_name, params):
        return dataset_name + "_UMAP_" + str(params['n_components']) + '_' + str(params['n_neighbors'])
