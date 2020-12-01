import pickle
import numpy
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datetime import datetime
from h5py import File

ROOT_DIR = "/home/blair/peat/preprocessing/decade_data/"
ROOT_DIR_P = "/home/blair/peat/preprocessing/sample_data/"
# TODO CWFIS
TEMPORAL_FT = {'VIIRS': 'VIIRS.h5', 
                'tCO2_1': 'CT2019B_flux_fire.h5', 
                'tCO2_2': 'CT2019B_flux_fuel.h5', 
                'CO2': 'CT2019B_XCO2_global_0.h5', 
                'CO2_b': 'CT2019B_XCO2_global_1.h5', 
                'CO2_c': 'CT2019B_XCO2_global_2.h5', 
                'CO2_d': 'CT2019B_XCO2_global_3.h5', 
                'CO2_e': 'CT2019B_XCO2_global_4.h5', 
                'CO2_f': 'CT2019B_XCO2_global_5.h5', 
                'CO2_g': 'CT2019B_XCO2_global_6.h5', 
                'CO2_h': 'CT2019B_XCO2_global_7.h5',
                'CO2_i': 'CT2019B_XCO2_global_8.h5', 
                'CO2_j': 'CT2019B_XCO2_global_9.h5', 
                'CO2_k': 'CT2019B_XCO2_global_10.h5', 
                'CWFIS': 'CWFIS.h5', 
                'ERA5': 'ERA5.h5'}
TEMPORAL_FT_P = {}
STATIC_FT = {'GSOC': 'GSOC.h5'}

# current shapes
'''
    CWFIS: (366, 4, 483, 910)
        # Burned Area Composites 
        'burned_pct_cat', 'burned_cause', 'fire_size_agency', 'fire_size_calc'

    GSOC: (483, 910) 
        # static soil carbon

    MODIS : (366, 3, 483, 910) 
        # burnt area hotspots
        'brightness', 'confidence', 'frp'
        --> use confidence for different losses

    # why is this different 
    TARNOCAI: (6, 483, 909) 
        # peat amount : 
        'PEATLAND_P', 'TOCC', 'BOG_PCT', 'FEN_PCT', 'SWAMP_PCT', 'MARSH_PCT'

    VIIRS : (366, 3, 483, 910)
        # burnt area hotspots
'''
def normalize(data):
    std = np.std(data)
    mean = np.mean(data)
    return (data-mean)/(std+0.01)

def lpickle(fname, index=None):
    with open(ROOT_DIR_P + fname + ".p", 'rb') as pickle_file:
        data = pickle.load(pickle_file)
        
    if index!=None:
        data = data[0][index]
    else:
        data = data[0]
    if fname[:5]=='CWFIS' and data.shape[-1]!=910:
        data = np.concatenate((data, data[-1, :][None,:]), axis=0)
        data = np.concatenate((data, data[:, -1][:, None]), axis=1)
    return np.nan_to_num(data)


def get_h5(file_name, check=False, index=None):
    h5_object = File(ROOT_DIR + file_name, 'r')
    data = h5_object.get('data')
    if check:
        time = h5_object.get('dates')
        time_obj = time[()].view('<M8[D]')
    else:
        time = None
        time_obj = None
    if index!=None:
        data = data[index]
    return data, time, time_obj

class PeatDataset(Dataset):

    def init_ft(self, in_features, out_ft):
        if not in_features:
            in_features = ['GSOC', 'VIIRS', 'TARNOCAI', 'CO2', 'ERA5', 'CWFIS']
        self.temp_ft = [t_ft for t_ft in TEMPORAL_FT]
        self.temp_ft_p = [t_ft for t_ft in TEMPORAL_FT_P]
        self.static_ft = [s_ft for s_ft in STATIC_FT]        
        self.out_ft = out_ft
        

    def __init__(self, pred_type='prediction', tarnocai_h=(0, 10), out_ft='cwfis', in_features=None, in_days=3, out_days=1, batch_size=1, train=True, model='cnn_lstm') :
        self.pred_type = pred_type
        self.init_ft(in_features, out_ft)
        self.in_days = in_days
        self.out_days = out_days
        tarnocai, _, _= get_h5('Tarnocai.h5')
        tarnocai = np.nan_to_num(np.array(tarnocai))
        max_start_time = -1
        static = []
        self.train = train
        self.num_static = 0
        
        if self.out_ft in TEMPORAL_FT:
            self.out, self.out_time, self.out_obj = get_h5(TEMPORAL_FT[self.out_ft], True)
            max_start_time = self.out_time[0]
        else:
            self.out_time =(0, None)
        
        self.all_ft = {}
        self.all_times = {}
        self.all_start = {}
        self.time_obj = {}
        self.num_temporal = 0
        self.temp_ft = [t_ft for t_ft in self.temp_ft if not(t_ft==out_ft) and not(t_ft=="CWFIS")]
        '''
        if self.pred_type=='prediction':
            self.temp_ft = [t_ft for t_ft in self.temp_ft if not(t_ft[:3] == 'CO2'[:3])]
        '''

        for t_ft in TEMPORAL_FT:
            self.all_ft[t_ft], self.all_times[t_ft], self.time_obj[t_ft] = (get_h5(TEMPORAL_FT[t_ft], True))
            if max_start_time<0:
                max_start_time = self.all_times[t_ft][0]
            else:
                max_start_time = max(self.all_times[t_ft][0], max_start_time)
            num_ft = self.all_ft[t_ft].shape[1] if len(self.all_ft[t_ft].shape)==4 else 1
            num_ft = num_ft if t_ft in self.temp_ft else 0
            self.num_temporal = self.num_temporal + num_ft

        for t_ft in TEMPORAL_FT:
            self.all_start[t_ft] = max_start_time- self.all_times[t_ft][0]

        for s_ft in self.static_ft:
            ft = np.nan_to_num(np.array(tarnocai[1]))
            static.append(ft)
            self.num_static = self.num_static + 1
        
        
        t_shape = self.all_ft[self.temp_ft[0]].shape
        self.num_total_days = t_shape[0]
        self.height = t_shape[-2]
        self.width = t_shape[-1]
        self.train_size = int(self.num_total_days)
        self.test_size = self.num_total_days - self.train_size
        static = np.stack(static, axis=0).astype('float32')[None, :]
        self.static = static
        peat_map = (tarnocai[0] > tarnocai_h[1]).reshape(1, 1, self.height, self.width)
        self.peat_map = np.ones(peat_map.shape) * peat_map
        self.out_start = max_start_time - self.out_time[0]

    def __len__(self):
        return (self.train_size - self.in_days - self.out_days)
        
    def _get_pickle_out_(self, start_date, num_days):
        ft = []
        for i in range(num_days):
            fname = self.out_ft + "_" + str(start_date + np.timedelta64(i,'D')) 
            if self.pred_type=='class':
                t_idx = lpickle(fname, index=0)
            else:
                t_idx = lpickle(fname, index=-1)
            ft.append(t_idx)
        ft_val = np.array(ft).astype('float32')
        out = np.array(ft_val)
        return out

    def _get_h5_out_(self, start_date, num_days):
        s_ft = self.all_start[self.out_ft] + start_date
        times = self.all_times[self.out_ft][()].view('<M8[D]')[s_ft: s_ft+num_days]
        if self.pred_type=='class':
            idx = 0
        else:
            idx = -1
        return np.nan_to_num((self.out[start_date: start_date+num_days, idx]))
    
    def _get_fire_out_(self, start_date, num_days):
        s_ft = self.all_start[self.out_ft] + start_date
        times = self.all_times[self.out_ft][()].view('<M8[D]')[s_ft: s_ft+num_days]
        if self.pred_type=='class':
            idx = 0
        else:
            idx = -1
        ft_val = self.all_ft['CWFIS'][t_idx: t_idx+self.in_days].astype('float32')
        times = self.all_times['CWFIS'][()].view('<M8[D]')[t_idx: t_idx+self.in_days]
        return np.nan_to_num(ft_val) > 0
        #np.nan_to_num((self.all_ft[t_ft][start_date: start_date+num_days, idx]))

    def _get_temp_ft_(self, idx):
        temporal = []
        # getting values in h5py
        k = 0
        for t_ft in self.temp_ft:
            t_idx = self.all_start[t_ft] + idx
            #print(t_idx, t_idx+self.in_days, t_ft)
            ft_val = self.all_ft[t_ft][t_idx: t_idx+self.in_days].astype('float32')
            times = self.all_times[t_ft][()].view('<M8[D]')[t_idx: t_idx+self.in_days]
            start_date = times[0]
            out_date = times[-1]
            ft_val = ft_val[:, :, :, :910]
            for i in range(ft_val.shape[1]):
                temporal.append(normalize(np.nan_to_num(ft_val[:, i])))
                k+=1
        # getting values in pickle format
        for t_ft in self.temp_ft_p:
            ft = []
            for i in range(self.in_days):
                fname = t_ft + " " + str(start_date +  np.timedelta64(i,'D'))
                t_idx = lpickle(fname)
                ft.append(t_idx)
            ft_val = np.array(ft).astype('float32')
            for i in range(ft_val.shape[1]):
                temporal.append(normalize(np.nan_to_num(ft_val[:, i])))
        return temporal, start_date, out_date

        

    def __getitem__(self, idx):
        temporal, start_date, out_date = self._get_temp_ft_(idx) 
        #fire_val = None
        if self.pred_type == 'prediction':
            if self.out_ft in TEMPORAL_FT_P:
                out_val = self._get_pickle_out_(out_date+np.timedelta64(1,'D'), self.out_days)
            else:
                out_val = self._get_h5_out_(idx + self.in_days, self.out_days)
            out_val =  (out_val - 405)/8.92 * self.peat_map.squeeze(1)
            #fire_val = self._get_fire_out_(idx + self.in_days, self.out_days)
            #out_val = normalize(out_val * self.peat_map.squeeze(1))

        elif self.pred_type == 'corr':
            if self.out_ft in TEMPORAL_FT_P:
                out_val = self._get_pickle_out_(start_date, self.out_days)
            else:
                out_val = self._get_h5_out_(idx, self.in_days)
            out_val = normalize(out_val * self.peat_map.squeeze(1))

        elif self.pred_type == 'class':
            if self.out_ft in TEMPORAL_FT_P:
                out_val = self._get_pickle_out_(out_date+np.timedelta64(1,'D'), self.out_days)
            else:
                out_val = self._get_h5_out_(idx + self.in_days, self.out_days)
            out_val = (out_val * self.peat_map.squeeze(1)).astype(int)//4
        temporal = np.stack(temporal, axis=0)
        static = self.static
        return temporal, self.static, out_val