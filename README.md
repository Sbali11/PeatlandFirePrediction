
Peat Fire Prediction Models

The following models are implemented in this directory:
1. Linear Regression
2. UNET
3. UNET_LSTM
4. GNN

Please refer to the paper for the reasons of why these models are selected. 

The directory is structured as follows-
- dataloader:
    - peat_loader: loader used for linear, unet, unet_lstm models
        returns the static and temporal features iteratively
        Please change the path at the top and provide the directory where the files are located 
    - peat_gnn : loader used for gnn based models
    provides different functions for adding different kinds of edges. Currently, only uses peat edges. Greater number of peat edges will require higher compute facilities

- model: contains all the relevant models. 
    - linear: regular linear regression based model
    - ConvBlock : Produces the downsampling and upsampling blocks for unet architectures
    - unet: A regular UNet Architecture
    - unet_lstm: unet augmented with a LSTM layer at the end
    - unet_gnn : model that first produces a gnn encoding and then produces the output
    - gnn : relational gnn model used to process differnt kinds of edges(requires high compute power)

train.py :
    train file for UNet, UNet-LSTM, Linear regression

train_gnn.py:
    train file for gnn based architecture


'''
Please follow the following steps to run the code:
1. Install required packages:
    - pytorch
    - pytorch geometric
    - skikit
    - matplotlib 
    - simplejson
2. python [train.py/train_gnn.py] --model <name of model> --conf f --dmodel <hyperparameter for num features> --lr <learning rate> --in_days <num of in days> --out_days <num of out days> --out <feature to output: CO2/CWFIS> 
'''
You can also pass in the hyperparameters in the form of a yaml file, in which case use --conf y --output_dir <name of directory with conf.yaml file>. This is useful in cases where you want to run a grid search. 


Follow the following steps to create a new model to test
The name of the model should be the same as the file name in Model

To add additional models, follow the following steps:
1. add file in model directory
2. name the class as Model
3. add code for the initialization in train.py/train_gnn.py
