
# Peat Fire Prediction Models

The following models are implemented in this directory:
1. Linear Regression
2. UNET
3. UNET_LSTM
4. GNN

Please refer to the paper for the reasons of why these models are selected. 

## Repository Structure
The directory is structured as follows-

### dataloader:
#### peat_loader: 
Loader used for linear, unet, unet_lstm models.
The loader contains the following Constant values: 
##### TEMPORAL_FT: 
these are all the temporal features used for the model prediction. However, the value to be predicted is always deleted before the processing. You can customize this to add/delete datasets for your own experiments
##### ROOT_DIR, ROOT_DIR_P
The dataloader supports both pickle and h5 files for the temporal features. The path for the h5 features needs to be provided in the ROOT_DIR and for the pickle features should be provided in the ROOT_DIR_P. Please change the path at the top and provide the directory where the files are located 

##### STATIC_FT: 
Currently the only static feature supported is tarnocai, but you can edit the STATIC_FT to add more features
The loaders also have the following functions and Classes:
* lpickle : to load the pickle files from filename fname
* get_h5 : read h5 files from file_name fname, you can also provide the optional argument index in case you only want a single index, the default values provided consist of all the features. Provide the optional argument check as True to read temporal files. This function returns the data(i.e the feature values), directly read dates, and dates in the day-month format. In case check is False, the last two are None

##### class PeatDataset: 
This takes in the hyperparameters for constructing the dataset. Iterating over this would provide the spatial, temporal, the ground truth output values when called returns the static and temporal features iteratively. 
The dataset consists of the folowing feature values that can be accessed from outside the class directly:
       - num_total_days: the total days in the h5 objects
       - static: the static features
       - peat_map : binary map of the peatlands constructed from the tarnocai file
       - num_temporal: the total number of temporal features
       - num_static: the total number of static features

#### peat_gnn: 
This is the pytorch geometric dataset for graph based models. All the functions are similar to the peat_loader file, except the fact that iterating over this will lead to different graphs instead of torch tensors. 
    - peat_gnn : loader used for gnn based models
    provides different functions for adding different kinds of edges. Currently, only uses peat edges. Greater number of peat edges will require higher compute facilities

### model: 
contains all the relevant models. 
#### linear: 
Regular Linear Regression based model. This model regards both spatial information in the batch dimension and each feature consists of both the temporal and spatial values. 
#### ConvBlock: 
This is a helper class that provides the downsampling and the upsampling blocks for the unet architectures
#### unet : 
A regular UNet Architecture consists of multiple downsampling blocks followed by upsampling blocks. 

#### unet_lstm: 
unet augmented with a LSTM layer at the end
#### unet_gnn
model that first produces a gnn encoding and then produces the output

#### gnn : 
relational gnn model used to process differnt kinds of edges(requires high compute power)

## train.py :
train file for UNet, UNet-LSTM, Linear regression

## train_gnn.py:
train file for gnn based architecture



Please follow the following steps to run the code:
> 1. Install required packages:
    - pytorch
    - pytorch geometric
    - skikit
    - matplotlib 
    - simplejson
> 2. `python [train.py/train_gnn.py] --model <name of model> --conf f --dmodel <hyperparameter for num features> --lr <learning rate> --in_days <num of in days> --out_days <num of out days> --out <feature to output: CO2/CWFIS> `

You can also pass in the hyperparameters in the form of a yaml file, in which case use --conf y --output_dir <name of directory with conf.yaml file>. This is useful in cases where you want to run a grid search. 


Follow the following steps to create a new model to test
The name of the model should be the same as the file name in Model

To add additional models, follow the following steps:
1. add file in model directory
2. name the class as Model
3. add code for the initialization in train.py/train_gnn.py
