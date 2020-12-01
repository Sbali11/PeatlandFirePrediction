'''
    Add 3 kinds of edges:
        spatial -> adjacent pixels
        spatial -> peat pixels 
        temporal

'''
import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
import pickle
import numpy
import numpy as np
from h5py import File
from collections import deque
import itertools
from torch_geometric.nn import knn_graph
from collections import defaultdict

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
        super().__init__()
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
        if self.pred_type=='class':
            self.temp_ft = [t_ft for t_ft in self.temp_ft if not(t_ft[:3] == 'CO2'[:3])]

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
        self.kw = self.width
        #//7
        #//7
        #//7
        #self.width
        self.kh = self.height
        #//7
        #//7
        #//3
        #print(self.kh * 7, self.kw * 7)
        #//3
        self.peat_edges()

    def get_one_d_idx(self, t, h, w, time_start, time_end, space_h, space_h_end, space_w, space_w_end):
        time_shift = (t-time_start) * self.kh * self.kw
        height_shift = (h-space_h) * self.kw 
        width_shift = (w-space_w)
        return time_shift + height_shift + width_shift

    def get_adj_edges(self, x, time_start, time_end, space_h, space_h_end, space_w, space_w_end):
        adj_edges_src = []
        adj_edges_dst = []
        edges_dict = defaultdict(list)
        all_nodes = np.arange(self.kh * self.kw * self.in_days).reshape(self.in_days, self.kh, self.kw)
        b_nodes = all_nodes[:, :-1].reshape(-1)
        t_nodes = all_nodes[:, 1:].reshape(-1)
        r_nodes = all_nodes[:, :, :-1].reshape(-1)
        l_nodes = all_nodes[:, :, 1:].reshape(-1)
        edges_src = np.concatenate((b_nodes, t_nodes, l_nodes, r_nodes))
        edges_dst = np.concatenate((t_nodes, b_nodes, r_nodes, l_nodes))
        #print("ADJ_EDGES", len(edges_src), len(edges_dst))

        return edges_src, edges_dst, edges_dict

    def get_time_edges(self, time_start, time_end, space_h, space_h_end, space_w, space_w_end):
        
            each datapoint is flattened
            such that newpoints = T_i * h * w + h_i * w + w_i
        #print("TIME_EDGES", len(time_edges_src), len(time_edges_dst))
        return time_edges_src, time_edges_dst, time_edges_types
    
    def len(self):
        size = (self.train_size - self.in_days - self.out_days)
        return (self.train_size - self.out_days) * (self.height//self.kh) * (self.width//self.kw)

    def get_all_connected(self, h, w, local_ones):
        # classic dfs: TODO change to smarter addition of links--> 
        # make weights inversely proportional to distance
        q = deque()
        q.append((h, w))
        while q:
            (h, w) = q.pop()
            if h<0 or w<0 or h>=self.height or w>=self.width:
                continue
            if (h, w) in local_ones:
                continue
            local_ones.add((h, w))
            for di, dj in list(itertools.product([-1, 0, 1], repeat=2)):
                if di==0 and dj==0:
                    continue
                i = h + di
                j = h + dj
                if((i, j)) not in local_ones:
                    q.append((i, j))

    def dfs(self, node, parent, res, visited, v, k=3): 
        if node in v:
            return []
        if node in visited:
            return visited[node]
        if self.peat_map[0][0][node[0]][node[1]] != 1:
            return []
        if node[0]>= self.height or node[0]<0:
            return []
        if node[1]>= self.width or node[1]<0:
            return []
        if (k < 0): 
            return []
        val = [node]
        children = []
        v.add(node)
        for di, dj in list(itertools.product([-1, 0, 1], repeat=2)):
            if di==0 and dj==0:
                continue
            ###print((node[0]+di, node[1]+dj))
            val += self.dfs((node[0]+di, node[1]+dj), node, res, visited, v, k-1)
        visited[node] = list(set(val))
        return list(set(val))
    
    def peat_edges(self):
        visited = {}
        self.p_edges = {}
        for h in range(self.height):
            for w in range(self.width):
                if self.peat_map[0][0][h][w] != 1:
                    continue
                v = set()
                pts = self.dfs((h, w), -1, [], visited, v) 
                self.p_edges[(h, w)] = pts

  
    def get_peat_edges(self, edges_dict, time_start, time_end, space_h, space_h_end, space_w, space_w_end):
        # add edges between adjacent peat lands
        peat_edges_src = []
        peat_edges_dst = []
        visited = set([])
        edges_dict = defaultdict(list)
        peat_map = self.peat_map[space_h:space_h_end, space_w:space_w_end].reshape(-1)
        one_d_peat_map = self.peat_map[space_h:space_h_end, space_w:space_w_end].reshape(-1)
        peat_points = peat_map[one_d_peat_map==1].reshape(-1, 1)
        peat_points = (self.kh * self.kw *  np.arange(self.in_days).reshape(1, -1)) + peat_points
        peat_points = peat_points.astype(int)
        correct_nodes = np.arange(self.kh * self.kw * self.in_days).reshape(-1)[peat_points]
        all_nodes = np.arange(self.kh * self.kw * self.in_days).reshape(self.in_days, self.kh, self.kw)
        peat_edges_src, peat_edges_dst =  [], []
        for k in range(2, 2+5):
            b_nodes = all_nodes[:, :-k].reshape(-1)
            t_nodes = all_nodes[:, k:].reshape(-1)
            r_nodes = all_nodes[:, :, :-k].reshape(-1)
            l_nodes = all_nodes[:, :, k:].reshape(-1)
            b_nodes = b_nodes[np.isin(b_nodes, correct_nodes)]
            t_nodes = t_nodes[np.isin(t_nodes, correct_nodes)]
            r_nodes = r_nodes[np.isin(r_nodes, correct_nodes)]
            l_nodes = l_nodes[np.isin(l_nodes, correct_nodes)]
            peat_edges_src.append(np.concatenate((b_nodes, t_nodes, l_nodes, r_nodes), axis=0))
            peat_edges_dst.append(np.concatenate((t_nodes, b_nodes, r_nodes, l_nodes), axis=0))

        peat_edges_src = np.concatenate(peat_edges_src).reshape(-1)
        peat_edges_dst = np.concatenate(peat_edges_dst).reshape(-1)
        #print("PEAT", (peat_edges_src.shape), len(peat_edges_dst))
        return peat_edges_src, peat_edges_dst      

    def get_edges(self, x, time_start, time_end, space_h, space_h_end, space_w, space_w_end):
        adj_edges_src, adj_edges_dst, edges_dict = self.get_adj_edges(x, time_start, time_end, space_h, space_h_end, space_w, space_w_end)
        #time_edges_src, time_edges_dst, time_edges_types = self.get_time_edges(time_start, time_end, space_h, space_h_end, space_w, space_w_end)
        peat_edges_src, peat_edges_dst = self.get_peat_edges(edges_dict, time_start, time_end, space_h, space_h_end, space_w, space_w_end)
        #print("ADJ", len(adj_edges_src), "TIME", len(time_edges_src), "PEAT", len(peat_edges_src))
        
        #all_edges_src = np.concatenate((time_edges_src, adj_edges_src, peat_edges_src))
        all_edges_src = np.concatenate((peat_edges_src, adj_edges_src))
        #all_edges_dst = np.concatenate((time_edges_dst, adj_edges_dst, peat_edges_dst))
        all_edges_dst = np.concatenate((peat_edges_dst, adj_edges_dst))
        #all_edges_dst =  adj_edges_dst +  peat_edges_dst
        #edge_types =  [0]*len(adj_edges_src) +  [1]*len(peat_edges_src)
        edge_types = np.concatenate((np.zeros(len(peat_edges_dst)), np.ones(len(adj_edges_dst))))
        #edge_types = np.concatenate((np.zeros(len(time_edges_src)), np.ones(len(adj_edges_src)), 1+np.ones(len(peat_edges_src))))
        #print(edge_types.shape, all_edges_src.shape, all_edges_dst.shape)
        return all_edges_src, all_edges_dst, edge_types
    
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

    def _get_h5_out_(self, start_date, num_days,  time_start, time_end, space_h, space_h_end, space_w, space_w_end):
        if self.pred_type == 'class':
            i = 0
        else:
            i = -1
        return np.nan_to_num((self.out[start_date: start_date+num_days, i, space_h:space_h_end, space_w:space_w_end]))

    def _get_temp_ft_(self, idx, time_start, time_end, space_h, space_h_end, space_w, space_w_end):
        temporal = []
        # getting values in h5py
        k = 0
        for t_ft in self.temp_ft:
            t_idx = self.all_start[t_ft] + idx
            ##print(t_idx, t_idx+self.in_days, t_ft)
            ft_val = self.all_ft[t_ft][t_idx: t_idx+self.in_days].astype('float32')
            times = self.all_times[t_ft][()].view('<M8[D]')[t_idx: t_idx+self.in_days]
            start_date = times[0]
            out_date = times[-1]

            ##print(t_ft, ft_val.shape)
            ft_val = ft_val[:, :, space_h:space_h_end, space_w:space_w_end]
            #[:, :, :, :910]
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


    def get(self, idx):
        temporal_features = []

        #time_shift = (t-time_start) * self.kh * self.kw
        #height_shift = (h-space_h) * self.kw 
        #width_shift = (w-space_w)

        space_w = (idx % (self.width//self.kw)) * self.kw
        idx = idx//(self.width //self.kw)
        space_h = (idx % (self.height//self.kh)) * self.kh
        idx = idx //((self.height//self.kh) * (self.width //self.kw))
        time_start = idx
        time_end = time_start + self.in_days 
        space_h_end = space_h + self.kh
        space_w_end = space_w + self.kw
        #print(space_w, space_w_end, space_h, space_h_end, time_start, time_end)
        if not self.train:
            time_start += self.train_size * k
        
        temporal_features, start_date, out_date = self._get_temp_ft_(idx, time_start, time_end, space_h, space_h_end, space_w, space_w_end)
        
        temporal_features = np.stack(temporal_features, axis=-1).astype('float32')
        # T, H, W, NUM_TEMP
        static_features = (self.static.reshape(1, 483, 910, 1))[:, space_h:space_h_end, space_w:space_w_end]
        # 1, H, W, NUM_TEMP
        static_features = np.repeat(static_features, self.in_days, axis=0)
        # T, H, W, NUM_STATIC
        node_features = np.concatenate([temporal_features, static_features], axis=-1)
        # T, H, W, NUM_FT
        #CHECK
        #node_features = node_features.permute(1, 2, 0, 3).reshape(self.kh, self.kw, -1)
        node_features = node_features.reshape(self.in_days * self.kh * self.kw , -1)
        # T * H * W, NUM_FT
        source_nodes, target_nodes, edge_types = self.get_edges(torch.tensor(node_features), time_start, time_end, space_h, space_h_end, space_w, space_w_end)
        #print("MAP ", self.peat_map.shape)
        peat_map = self.peat_map[:, :,  space_h:space_h_end, space_w:space_w_end]
        if self.pred_type == 'prediction':
            if self.out_ft in TEMPORAL_FT_P:
                out_val = self._get_pickle_out_(out_date+np.timedelta64(1,'D'), self.out_days)
            else:
                out_val = self._get_h5_out_(self.out_start + idx + self.in_days, self.out_days,  time_start, time_end, space_h, space_h_end, space_w, space_w_end)
            out_val = normalize(out_val * peat_map.squeeze(1))

        elif self.pred_type == 'corr':
            if self.out_ft in TEMPORAL_FT_P:
                out_val = self._get_pickle_out_(start_date, self.out_days)
            else:
                out_val = self._get_h5_out_(self.out_start + idx, self.in_days,  time_start, time_end, space_h, space_h_end, space_w, space_w_end)
            out_val = normalize(out_val * peat_map.squeeze(1))

        elif self.pred_type == 'class':
            if self.out_ft in TEMPORAL_FT_P:
                out_val = self._get_pickle_out_(out_date+np.timedelta64(1,'D'), self.out_days)
            else:
                out_val = self._get_h5_out_(self.out_start + idx + self.in_days, self.out_days,  time_start, time_end, space_h, space_h_end, space_w, space_w_end)
            out_val = (out_val * peat_map.squeeze(1)).astype(int)//4

        #out_val = out_val.transpose(1, 2, 0) # time steps at end
        
        x = torch.tensor(node_features)
        #print("X", x.shape, "TEMPORAL", temporal_features.shape)
        #exit(0)
        y = torch.tensor(out_val)
        edge_types = torch.LongTensor(edge_types)
        edge_index = torch.LongTensor([source_nodes, target_nodes])
        data = Data(x=x, edge_index=edge_index, y=y, edge_types=edge_types, 
                    num_nodes=x.shape[0], out_days=self.out_days, 
                    num_output_nodes=y.shape[0], num_relations=2, 
                    in_days=self.in_days, peat_map=torch.FloatTensor(peat_map))

        return data

    

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
        

    def __init__(self, temporal_in=False, pred_type='prediction', tarnocai_h=(0, 10), out_ft='cwfis', in_features=None, in_days=3, out_days=1, batch_size=1, train=True, model='cnn_lstm') :
        super().__init__()
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
        self.temporal_in = temporal_in
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
        self.kw = self.width
        #//7
        #//7
        #//7
        #self.width
        self.kh = self.height
        #//7
        #//7
        #//3
        #print(self.kh * 7, self.kw * 7)
        #//3
        self.get_peat_edges()
        self.get_adj_edges()

    def get_one_d_idx(self, t, h, w, time_start, time_end, space_h, space_h_end, space_w, space_w_end):
        time_shift = (t-time_start) * self.kh * self.kw
        height_shift = (h-space_h) * self.kw 
        width_shift = (w-space_w)
        return time_shift + height_shift + width_shift

    def get_adj_edges(self):
        #, x, time_start, time_end, space_h, space_h_end, space_w, space_w_end):
        adj_edges_src = []
        adj_edges_dst = []
        edges_dict = defaultdict(list)
        all_nodes = np.arange(self.kh * self.kw).reshape(1, self.kh, self.kw)
        b_nodes = all_nodes[:, :-1].reshape(-1)
        t_nodes = all_nodes[:, 1:].reshape(-1)
        r_nodes = all_nodes[:, :, :-1].reshape(-1)
        l_nodes = all_nodes[:, :, 1:].reshape(-1)
        #self.adj_edges_src, self.adj_edges_dst
        self.adj_edges_src = np.concatenate((b_nodes, t_nodes, l_nodes, r_nodes))
        self.adj_edges_dst = np.concatenate((t_nodes, b_nodes, r_nodes, l_nodes))
        #print("ADJ_EDGES", len(edges_src), len(edges_dst))

        #return edges_src, edges_dst, edges_dict

    def get_time_edges(self, time_start, time_end, space_h, space_h_end, space_w, space_w_end):
        '''
            each datapoint is flattened
            such that newpoints = T_i * h * w + h_i * w + w_i
        '''
        #print("TIME_EDGES", len(time_edges_src), len(time_edges_dst))
        return time_edges_src, time_edges_dst, time_edges_types
    
    def len(self):
        size = (self.train_size - self.in_days - self.out_days)
        return (self.train_size - self.in_days- self.out_days) * (self.height//self.kh) * (self.width//self.kw)

    def get_all_connected(self, h, w, local_ones):
        # classic dfs: TODO change to smarter addition of links--> 
        # make weights inversely proportional to distance
        q = deque()
        q.append((h, w))
        while q:
            (h, w) = q.pop()
            if h<0 or w<0 or h>=self.height or w>=self.width:
                continue
            if (h, w) in local_ones:
                continue
            local_ones.add((h, w))
            for di, dj in list(itertools.product([-1, 0, 1], repeat=2)):
                if di==0 and dj==0:
                    continue
                i = h + di
                j = h + dj
                if((i, j)) not in local_ones:
                    q.append((i, j))

    def dfs(self, node, parent, res, visited, v, k=3): 
        if node in v:
            return []
        if node in visited:
            return visited[node]
        if self.peat_map[0][0][node[0]][node[1]] != 1:
            return []
        if node[0]>= self.height or node[0]<0:
            return []
        if node[1]>= self.width or node[1]<0:
            return []
        if (k < 0): 
            return []
        val = [node]
        children = []
        v.add(node)
        for di, dj in list(itertools.product([-1, 0, 1], repeat=2)):
            if di==0 and dj==0:
                continue
            ###print((node[0]+di, node[1]+dj))
            val += self.dfs((node[0]+di, node[1]+dj), node, res, visited, v, k-1)
        visited[node] = list(set(val))
        return list(set(val))
    
    def peat_edges(self):
        visited = {}
        self.p_edges = {}
        for h in range(self.height):
            for w in range(self.width):
                if self.peat_map[0][0][h][w] != 1:
                    continue
                v = set()
                pts = self.dfs((h, w), -1, [], visited, v) 
                self.p_edges[(h, w)] = pts

  
    def get_peat_edges(self):
        #, edges_dict, time_start, time_end, space_h, space_h_end, space_w, space_w_end):
        # add edges between adjacent peat lands
        peat_edges_src = []
        peat_edges_dst = []
        visited = set([])
        edges_dict = defaultdict(list)
        peat_map = self.peat_map.reshape(-1)
        #[space_h:space_h_end, space_w:space_w_end].reshape(-1)
        one_d_peat_map = self.peat_map.reshape(-1)
        #[space_h:space_h_end, space_w:space_w_end].reshape(-1)
        peat_points = peat_map[one_d_peat_map==1].reshape(-1)
        #peat_points = 
        #(self.kh * self.kw *  np.arange(self.in_days).reshape(1, -1)) + peat_points
        peat_points = peat_points.astype(int)
        correct_nodes = np.arange(self.kh * self.kw).reshape(-1)[one_d_peat_map==1]
        #print(correct_nodes, correct_nodes.shape)
        point_coords = np.arange(self.kh * self.kw)
        #peat_map = self.peat_map(self.kh * self.kw)
        '''
        comb_array = np.array(np.meshgrid(point_coords, point_coords)).T.reshape(-1, 2).T  # find all possible combinations
        pt1 = comb_array[0]
        pt2 = comb_array[1]
        comb_array = comb_array[numpy.linalg.norm(comb_array[0]- comb_array[1]) <= 4] 
        comb_array = comb_array[comb_array[0].isin(correct_nodes) and comb_array[1].isin(correct_nodes)]
        self.peat_edges_src = np.concatenate((comb_array[0], comb_array[1]), axis=0).reshape(-1)
        self.peat_edges_dst = np.concatenate((comb_array[1], comb_array[0]), axis=0).reshape(-1)
        '''



        #print("PEATLANDS", correct_nodes.shape)
        all_nodes = np.arange(self.kh * self.kw).reshape(1, self.kh, self.kw)
        peat_edges_src, peat_edges_dst =  [], []
        #CHECK
        for k in range(2, 2+50):
            b_nodes = all_nodes[:, :-k].reshape(-1)
            t_nodes = all_nodes[:, k:].reshape(-1)
            r_nodes = all_nodes[:, :, :-k].reshape(-1)
            l_nodes = all_nodes[:, :, k:].reshape(-1)
            vertical = np.isin(b_nodes, correct_nodes) * np.isin(t_nodes, correct_nodes)
            horizontal = np.isin(r_nodes, correct_nodes) * np.isin(l_nodes, correct_nodes)
            b_nodes = b_nodes[vertical]
            t_nodes = t_nodes[vertical]
            r_nodes = r_nodes[horizontal]
            l_nodes = l_nodes[horizontal]
            peat_edges_src.append(np.concatenate((b_nodes, t_nodes, l_nodes, r_nodes), axis=0))
            peat_edges_dst.append(np.concatenate((t_nodes, b_nodes, r_nodes, l_nodes), axis=0))

        self.peat_edges_src = np.concatenate(peat_edges_src).reshape(-1)
        self.peat_edges_dst = np.concatenate(peat_edges_dst).reshape(-1)
        #print("PEAT", (peat_edges_src.shape), len(peat_edges_dst))
        
        #return peat_edges_src, peat_edges_dst      

    def get_edges(self, x, time_start, time_end, space_h, space_h_end, space_w, space_w_end):
        adj_edges_src, adj_edges_dst = np.array([]), np.array([])
        #self.adj_edges_src, self.adj_edges_dst
        #self.get_adj_edges(x, time_start, time_end, space_h, space_h_end, space_w, space_w_end)
        #time_edges_src, time_edges_dst, time_edges_types = self.get_time_edges(time_start, time_end, space_h, space_h_end, space_w, space_w_end)
        peat_edges_src, peat_edges_dst = self.peat_edges_src, self.peat_edges_dst
        #self.get_peat_edges(edges_dict, time_start, time_end, space_h, space_h_end, space_w, space_w_end)
        #print("ADJ", len(adj_edges_src), "TIME", len(time_edges_src), "PEAT", len(peat_edges_src))
        
        #all_edges_src = np.concatenate((time_edges_src, adj_edges_src, peat_edges_src))
        all_edges_src = np.concatenate((peat_edges_src, adj_edges_src))
        #all_edges_dst = np.concatenate((time_edges_dst, adj_edges_dst, peat_edges_dst))
        all_edges_dst = np.concatenate((peat_edges_dst, adj_edges_dst))
        #all_edges_dst =  adj_edges_dst +  peat_edges_dst
        #edge_types =  [0]*len(adj_edges_src) +  [1]*len(peat_edges_src)
        edge_types = np.concatenate((np.zeros(len(peat_edges_dst)), np.ones(len(adj_edges_dst))))
        #edge_types = np.concatenate((np.zeros(len(time_edges_src)), np.ones(len(adj_edges_src)), 1+np.ones(len(peat_edges_src))))
        #print(edge_types.shape, all_edges_src.shape, all_edges_dst.shape)
        return all_edges_src, all_edges_dst, edge_types
    
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

    def _get_h5_out_(self, start_date, num_days,  time_start, time_end, space_h, space_h_end, space_w, space_w_end):
        if self.pred_type == 'class':
            i = 0
        else:
            i = -1
        return np.nan_to_num((self.out[start_date: start_date+num_days, i, space_h:space_h_end, space_w:space_w_end]))

    def _get_temp_ft_(self, idx, time_start, time_end, space_h, space_h_end, space_w, space_w_end):
        temporal = []
        # getting values in h5py
        k = 0
        for t_ft in self.temp_ft:
            t_idx = self.all_start[t_ft] + idx
            ##print(t_idx, t_idx+self.in_days, t_ft)
            ft_val = self.all_ft[t_ft][t_idx: t_idx+self.in_days].astype('float32')
            times = self.all_times[t_ft][()].view('<M8[D]')[t_idx: t_idx+self.in_days]
            start_date = times[0]
            out_date = times[-1]

            ##print(t_ft, ft_val.shape)
            ft_val = ft_val[:, :, space_h:space_h_end, space_w:space_w_end]
            #[:, :, :, :910]
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


    def get(self, idx):
        temporal_features = []
        '''
        time_shift = (t-time_start) * self.kh * self.kw
        height_shift = (h-space_h) * self.kw 
        width_shift = (w-space_w)
        '''
        space_w = (idx % (self.width//self.kw)) * self.kw
        idx = idx//(self.width //self.kw)
        space_h = (idx % (self.height//self.kh)) * self.kh
        idx = idx //((self.height//self.kh) * (self.width //self.kw))
        time_start = idx
        time_end = time_start + self.in_days 
        space_h_end = space_h + self.kh
        space_w_end = space_w + self.kw
        #print(space_w, space_w_end, space_h, space_h_end, time_start, time_end)
        if not self.train:
            time_start += self.train_size * k
        
        temporal_features, start_date, out_date = self._get_temp_ft_(idx, time_start, time_end, space_h, space_h_end, space_w, space_w_end)
        temporal_ret = temporal_features
        #.clone()
        temporal_features = np.stack(temporal_features, axis=-1).astype('float32')
        # T, H, W, NUM_TEMP
        static_features = (self.static.reshape(1, 483, 910, 1))[:, space_h:space_h_end, space_w:space_w_end]
        # 1, H, W, NUM_TEMP
        static_features = np.repeat(static_features, self.in_days, axis=0)
        # T, H, W, NUM_STATIC
        node_features = np.concatenate([temporal_features, static_features], axis=-1)
        # T, H, W, NUM_FT
        #CHECK
        if self.temporal_in:
            node_features = torch.tensor(node_features).permute(1, 2, 0, 3).reshape(self.kh * self.kw * self.in_days, -1)
        else:
            node_features = torch.tensor(node_features).permute(1, 2, 0, 3).reshape(self.kh * self.kw, -1)
        # T * H * W, NUM_FT
        source_nodes, target_nodes, edge_types = self.get_edges(torch.tensor(node_features), time_start, time_end, space_h, space_h_end, space_w, space_w_end)
        #print("MAP ", self.peat_map.shape)
        peat_map = self.peat_map[:, :,  space_h:space_h_end, space_w:space_w_end]
        if self.pred_type == 'prediction':
            if self.out_ft in TEMPORAL_FT_P:
                out_val = self._get_pickle_out_(out_date+np.timedelta64(1,'D'), self.out_days)
            else:
                out_val = self._get_h5_out_(self.out_start + idx + self.in_days, self.out_days,  time_start, time_end, space_h, space_h_end, space_w, space_w_end)
            out_val =  (out_val - 405)/8.92 * self.peat_map.squeeze(1)
            #normalize(out_val * peat_map.squeeze(1))

        elif self.pred_type == 'corr':
            if self.out_ft in TEMPORAL_FT_P:
                out_val = self._get_pickle_out_(start_date, self.out_days)
            else:
                out_val = self._get_h5_out_(self.out_start + idx, self.in_days,  time_start, time_end, space_h, space_h_end, space_w, space_w_end)
            out_val = normalize(out_val * peat_map.squeeze(1))

        elif self.pred_type == 'class':
            if self.out_ft in TEMPORAL_FT_P:
                out_val = self._get_pickle_out_(out_date+np.timedelta64(1,'D'), self.out_days)
            else:
                out_val = self._get_h5_out_(self.out_start + idx + self.in_days, self.out_days,  time_start, time_end, space_h, space_h_end, space_w, space_w_end)
            out_val = (out_val * peat_map.squeeze(1)).astype(int)//4

        x = torch.FloatTensor(node_features)
        y = torch.LongTensor(out_val)
        edge_types = torch.LongTensor(edge_types)
        edge_index = torch.LongTensor([source_nodes, target_nodes])
        temporal_ret = torch.tensor(np.stack(temporal_ret, axis=0))
        num_nodes = x.shape[0]
        data = Data(x=x, edge_index=edge_index, y=y, edge_types=edge_types, 
                    num_nodes=x.shape[0], out_days=self.out_days, 
                    static_ft=torch.tensor(self.static), 
                    temporal_ft=temporal_ret,
                    num_output_nodes=y.shape[0], num_relations=2, 
                    num_temporal=self.num_temporal, num_static=self.num_static,
                    in_days=self.in_days, peat_map=torch.FloatTensor(peat_map))

        return data
