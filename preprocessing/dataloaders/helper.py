from geocube.api.core import make_geocube
import pandas as pd
import rasterio as rio
import numpy as np

def get_numpy(feature, shapefile, resolution=0.1):
    grid = make_geocube(
        vector_data=shapefile,
        measurements=[feature],
        resolution=(-resolution, resolution),
    )

    slide = grid[feature]
    return slide

def get_affine(layer, feature):
    file_name = "{}.tif".format(feature)
    layer.rio.to_raster(file_name)
    with rio.open(file_name) as img:
        imgmeta = img.meta

    # https://datacube-core.readthedocs.io/en/latest/_modules/affine.html
    affine = imgmeta["transform"]
    affine_tuple = (affine.a, affine.b, affine.c, affine.d, affine.e, affine.f)
    return affine_tuple

def get_bounds(layer, affine_tuple):
    # Canada
    x1 = -141.0000
    y1 = 41.7500
    x2 = -50.0000
    y2 = 90.0000

    def lat_lon_to_ind(affine):
      x_cvt, _, x0, _, y_cvt, y0 = affine
      x1_ = (x1 - x0) / x_cvt
      y1_ = (y1 - y0) / y_cvt
      x2_ = (x2 - x0) / x_cvt
      y2_ = (y2 - y0) / y_cvt
      return (y2_, y1_, x1_, x2_)

    n_,s_,w_,e_ = lat_lon_to_ind(affine_tuple)
    return (n_,s_,w_,e_)

def bound_numpy(slide, bounds):
    n_, s_, w_, e_ = bounds
    
    block_height = int(s_ - n_)
    block_width = int(e_ - w_)
    block_y_start = max(0,-int(n_))
    block_x_start = max(0,-int(w_))
    
    (height, width) = slide.shape
    y_start = max(0, int(n_))
    x_start = max(0, int(w_))
    
    y_extent = min(height - y_start, block_height - block_y_start)
    x_extent = min(width - x_start, block_width - block_x_start)

    block = np.full((block_height, block_width), np.nan)
    block[block_y_start:block_y_start+y_extent,block_x_start:block_x_start+x_extent] = \
        slide[y_start:y_start+y_extent,x_start:x_start+x_extent]
    return block
