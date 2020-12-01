import numpy as np
import pyvips
from . import helper
from load_data import GenericLoader

class GSOCLoader(GenericLoader):
    def __init__(self):
        GenericLoader.__init__(self)
        # set these variables
        self.spatial_res = 0.00833333334
        self.static = True
        self.dataset_name = "GSOC"
        
        path = "/DataSet/peat/GSOC/carbon.tif"
        feature = "carbon"
        
        affine = (0.0083333333, 0.0, -180.0, 0.0, -0.008333333299989214, 83.616081247) # pre-computed from fixed spatial-res
        layer = pyvips.Image.new_from_file(path)
        A = np.ndarray(buffer=layer.write_to_memory(), dtype=np.float32, shape=[layer.height, layer.width])
        A[A < 0] = np.nan
        bounds = helper.get_bounds(A, affine) #n_,s_,w_,e_ 
        slide = helper.bound_numpy(A, bounds)
        self.data = slide
        self.features = [feature]
    
    def load_data(self):
        return (self.data, None, self.features)
    
    def load_date_data(self, date='2020-10-20'):
        return self.data