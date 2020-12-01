import datetime
import interpolate
import numpy as np
import pdb
import pickle
import time
import os
import yaml
from dataloaders.hotspot_loader import *
from dataloaders.CT2009B_loader import *
from dataloaders.CWFIS_loader import *
from dataloaders.ERA5_loader import *
from dataloaders.GSOC_loader import *
from dataloaders.SMAP_loader import *
from dataloaders.tarnocai_loader import *

        
class DataLoader():
    def __init__(self, spatial_res = 0.25):
        # self.config = yaml.safe_load(open('datasets.yaml'))
        self.spatial_res = spatial_res
        self.dataloaders = [VIIRSLoader, MODISLoader, CWFISLoader, ERA5Loader, GSOCLoader, TarnocaiLoader]
        self.cached = {}
        self.discover_cached()

#     def load_variable(self, month, year, variable):
#         if variable not in self.var_to_dataset:
#             raise ValueError(f'Variable {variable} not valid.')
#         return self.load_dataset(self.var_to_dataset[variable], month, year, variable)

#     def load_dataset(self, dataset, month, year, variable = None):
#         if dataset not in self.config['datasets']:
#             raise ValueError(f'Dataset {dataset} not valid.')
#         if variable is not None:
#             data = eval(f'self.load_{dataset}({month}, {year}, variable = \'{variable}\')')
#         else:
#             data = eval(f'self.load_{dataset}({month}, {year})')
#         source_spatial_res = self.config['datasets'][dataset]['spatial_resolution']
        
#         if len(data.shape)>=3:
#             for i, idx in enumerate(itertools.product(*[range(s) for s in data.shape[:-2]])):
#               tmp = interpolate.change_spatial_res(data[idx], source_spatial_res, self.spatial_res)
#               if i==0:
#                 int_data=np.zeros(data.shape[:-2]+tmp.shape)
#               int_data[idx]=tmp
#         else:
#             int_data = interpolate.change_spatial_res(data, source_spatial_res, self.spatial_res)
#         return int_data
    def discover_cached(self):
        for filename in os.listdir('.'):
            if filename.endswith(".p"): 
                self.cached[filename[:-2]] =  1
        
    def init_dataloaders(self):
        for i in range(len(self.dataloaders)):
            if (("dataloader_" + self.dataloaders[i].__name__)in self.cached):
                self.dataloaders[i] = self.load_pickled("dataloader_" + self.dataloaders[i].__name__)
                print(f'finished loading cached {type(self.dataloaders[i]).__name__}')
            else:
                self.dataloaders[i] = self.dataloaders[i]()
                print(f'finished loading {type(self.dataloaders[i]).__name__}')
                self.cache_data("dataloader_" + type(self.dataloaders[i]).__name__, self.dataloaders[i])
                self.cached["dataloader_" + type(self.dataloaders[i]).__name__] = 1

    def cache_data(self, name, data_dict):
        pickle.dump(data_dict, open(name + ".p", "wb"))
        
    def load_pickled(self, name):
        data_dict = pickle.load(open(name + ".p", "rb" ))
        return data_dict
    
    def load_loader(self, loader):
        filename = "data_" + type(loader).__name__
        if (filename in self.cached):
            print(f'loading cached {type(loader).__name__} data')
            return self.load_pickled(filename)
        else:
            print(f'loading {type(loader).__name__} data')
            data = loader.load_data()
            self.cache_data(filename, data)
            self.cached[filename] = 1
            return data

    def load_all_data(self, cache=True):
        datas, all_dates, all_features = zip(*[self.load_loader(dl) for dl in self.dataloaders])
        features = []
        for i, set_features in enumerate(all_features):
            features += [(self.dataloaders[i].dataset_name + feature) for feature in set_features]
        same_dates = sorted(list(set(all_dates[0]).intersection(*all_dates)))
        
        unified_data = []
        for i, data in enumerate(datas):
            start_idx = dates[i].index(same_dates[0])
            end_idx = dates[i].index(same_dates[-1])
            unified_data.append(data[start_idx:end_idx + 1])
        
        data, dates, features = (np.concatenate(unified_data, axis=1), same_dates, features)
        if (cache):
            data_dict = {
                "data": data,
                "dates": dates,
                "features": features
            }
            cache_data("all_data", data_dict)
        # concatenate by feature
        return (data, dates, features)
        