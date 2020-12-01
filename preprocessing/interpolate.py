import numpy as np
from pyresample.geometry import GridDefinition, SwathDefinition
from pyresample.kd_tree import resample_nearest

# Southeast Asia
# N_BOUND = 7.25 # lat degrees
# E_BOUND = 120.25 # lon degrees
# S_BOUND = -6.75 # lat degrees
# W_BOUND = 94.25 # lon degrees

# World
# N_BOUND = 90 # lat degrees
# E_BOUND = 180 # lon degrees
# S_BOUND = -90 # lat degrees
# W_BOUND = -180 # lon degrees

# Canada
W_BOUND = -141.0000
E_BOUND = -50.0000
N_BOUND = 90.0000
S_BOUND = 41.7500

def get_grid_latlng(spatial_res, offset = (0, 0)):
    if (type(spatial_res) is tuple):
        lat_res, lon_res = spatial_res
    else:
        lat_res = spatial_res
        lon_res = spatial_res
    lats = np.flip(np.arange(S_BOUND, N_BOUND + offset[0] * lat_res, lat_res))
    lons = np.arange(W_BOUND, E_BOUND + offset[1] * lon_res, lon_res)
    
    return get_latlng(lats, lons)

def get_latlng(lats, lons):
    grid_shape = lats.shape + lons.shape
    lats = np.broadcast_to(lats[...,None], grid_shape)
    lons = np.broadcast_to(lons, grid_shape)
    return (lats, lons)

def change_spatial_res(source_data, source_spatial_res, target_spatial_res, offset=(0,0)):
    source_lats, source_lons = get_grid_latlng(source_spatial_res, offset)
    print(source_lats.shape)
    return latlng_interpolate(source_data, source_lats, source_lons, target_spatial_res)

def latlng_interpolate(source_data, source_lats, source_lons, spatial_res):
    target_lats, target_lons = get_grid_latlng(spatial_res)
    if (len(source_data.shape) > 2):
        source_data = np.moveaxis(source_data, [-2, -1], [0, 1])

    source_grid = SwathDefinition(lons=source_lons, lats=source_lats)
    target_grid = GridDefinition(lons=target_lons, lats=target_lats)
    target_data = resample_nearest(source_grid, source_data, target_grid, 100000000.0)
    if (len(target_data.shape) > 2):
        target_data = np.moveaxis(target_data, [0, 1], [-2, -1])
    return target_data

# 
#
#