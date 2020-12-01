import numpy as np
import datetime

# API for Dataloader loaders
class GenericLoader():
    def __init__(self):
        # utility variables
        self.W_LIMIT = -141.0000
        self.E_LIMIT = -50.0000
        self.N_LIMIT = 90.0000
        self.S_LIMIT = 41.7500
        # set these variables
        self.spatial_res = 0.01
        self.static = False
        self.dataset_name = "GenericData"
        # do any initialization
        
    def load_data(self):
        # return data as numpy array with shape
        # (data x features x latitude x longitude)
        # if static, (features x latitude x longitude)
        # West, -141.0000; East, -50.0000; North, 90.0000; South, 41.7500
        # input list of features
        dates = self.get_date_range('2020-10-20', '2020-10-20')
        data = np.zeros((1,1,1,1))
        features = ['blank']
        return (data, dates, features)
    
    def load_date_data(self, date='2020-10-20'):
        # return data as numpy array with shape
        # (features x latitude x longitude)
        # West, -141.0000; East, -50.0000; North, 90.0000; South, 41.7500
        data = np.zeros((1,1,1))
        features = ['blank']
        return data, features
        
    def get_date_range(self, start, end):
        # get dates [start,...,end] (excusive) as accpted by load_data
        date_format = '%Y-%m-%d'
        start_time = datetime.datetime.strptime(start, date_format)
        end_time = datetime.datetime.strptime(end, date_format)
        days = (end_time - start_time).days + 1
        dates = [start_time + datetime.timedelta(days=i) for i in range(days)]
        return [date.strftime(date_format) for date in dates]