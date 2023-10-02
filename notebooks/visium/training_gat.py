#%%
import torch
import pandas as pd

from torch_geometric.loader import DataLoader

import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle   
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger

from torch.utils.data import Dataset
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import seaborn as sns
from tqdm import tqdm
import scanpy as sc
import spaceTree.utils as utils
from spaceTree.models import *
import os
import matplotlib as mpl

os.chdir("/home/o313a/clonal_GNN/")

#%%

data = torch.load("data/processed/data.pt")
with open('data/processed/full_encoding.pkl', 'rb') as handle:
    encoder_dict = pickle.load(handle)
node_encoder_rev = {val:key for key,val in encoder_dict["nodes"].items()}
node_encoder_clone = {val:key for key,val in encoder_dict["clones"].items()}
node_encoder_ct = {val:key for key,val in encoder_dict["types"].items()}

data.edge_attr = data.edge_attr.reshape((-1,1))
# %%
# Separate training data (scRNA) from spatial data
hold_out_indices = np.where(data.y_clone == -1)[0]
hold_out = torch.tensor(hold_out_indices, dtype=torch.long)

total_size = data.x.shape[0] - len(hold_out)
train_size = int(0.8 * total_size)

# Get indices that are not in hold_out
hold_in_indices = np.arange(data.x.shape[0])
hold_in = [index for index in hold_in_indices if index not in hold_out]

# %%
# Split the data into train and test sets
train_indices, test_indices, _, _ = train_test_split(
    hold_in, 
    data.y_clone[hold_in], 
    stratify=data.y_clone[hold_in], 
    test_size=0.2, 
    random_state=42
)

# Assign the indices to data masks
data.train_mask = torch.tensor(train_indices, dtype=torch.long)
data.test_mask = torch.tensor(test_indices, dtype=torch.long)

# Set the hold_out data
data.hold_out = hold_out


# %%
def compute_class_weights(y_train):
    """Calculate class weights based on the class sample count."""
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    return 1. / class_sample_count

# Calculate weights for 'y_type'
y_train_type = data.y_type[data.train_mask]
weight_type_values = compute_class_weights(y_train_type)
weight_type = torch.tensor(weight_type_values, dtype=torch.float)

# Calculate weights for 'y_clone'
y_train_clone = data.y_clone[data.train_mask]
weight_clone_values = compute_class_weights(y_train_clone)
weight_clone = torch.tensor(weight_clone_values, dtype=torch.float)

# %%
data.num_classes_clone = len(data.y_clone.unique())
data.num_classes_type = len(data.y_type.unique())

# %%
class GraphDataset(Dataset):
    """Custom Dataset for loading a single graph and its associated mask."""
    
    def __init__(self, data, mask):
        self.data = data
        self.mask = mask
    
    def __len__(self):
        # Returns 1 because this dataset contains only a single graph
        return 1  
    
    def __getitem__(self, idx):
        """Returns the graph data and its mask."""
        return self.data, self.mask

# Create dummy datasets for training and testing
train_dataset = GraphDataset(data, data.train_mask)
test_dataset = GraphDataset(data, data.test_mask)

# Create DataLoaders
# Note: Since the dataset contains only one graph, shuffling won't have an effect
train_loader = DataLoader(train_dataset, batch_size=1)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# %%

norm_sim = np.load("data/interim/clone_dist_over.npy")
sns.clustermap(norm_sim, figsize=(5,5))
norm_sim = torch.tensor(norm_sim)
# %%
device = torch.device('cuda:0')
data = data.to(device)
weight_clone = weight_clone.to(device)
weight_type = weight_type.to(device)
norm_sim = norm_sim.to(device)
#%%
import itertools

lrs = [1e-2,5e-2,1e-3,1e-4]
hid_dims = [32,64,128]
heads = [1,3,5]
all_params = [lrs, hid_dims, heads]
all_params = list(itertools.product(*all_params))
# %%
for lr, hid_dim, head in all_params:

    model = GATLightningModule(data, weight_clone, weight_type, norm_sim = norm_sim, learning_rate=lr, heads=head, dim_h = hid_dim)
    model = model.to(device)
    logger1 = TensorBoardLogger('logs_visium', name = f"round1_{lr}_{hid_dim}_{head}")
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="validation_combined_loss", min_delta=0.001, patience=10, verbose=True, mode="min")
    # Train
    trainer1 = pl.Trainer(max_epochs=1000, devices=1, accelerator = "cuda", logger=logger1, callbacks = [early_stop_callback], log_every_n_steps=10)
    trainer1.fit(model, train_loader, test_loader)

    # Switch to unweighted loss and reset the early stopping callback's state
    model.use_weighted = True
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="validation_combined_loss", min_delta=0.001, patience=10, verbose=True, mode="min")
    logger2 = TensorBoardLogger('logs_visium', name = f"round2_{lr}_{hid_dim}_{head}")

    # Train with unweighted loss
    trainer2 = pl.Trainer(max_epochs=1000, devices=1, accelerator = "cuda", logger=logger2, callbacks = [early_stop_callback],log_every_n_steps=60)
    trainer2.fit(model, train_loader, test_loader)
    del model
# %%
lr = 1e-2
hid_dim = 64
head = 1
model = GATLightningModule(data, weight_clone, weight_type, norm_sim = norm_sim, learning_rate=0.01, heads=3, dim_h = 32)
model = model.to(device)
logger1 = TensorBoardLogger('logs_visium', name = f"round1_{lr}_{hid_dim}_{head}")
early_stop_callback = pl.callbacks.EarlyStopping(monitor="validation_combined_loss", min_delta=0.001, patience=10, verbose=True, mode="min")
# Train
trainer1 = pl.Trainer(max_epochs=1000, devices=1, accelerator = "cuda", logger=logger1, callbacks = [early_stop_callback], log_every_n_steps=10)
trainer1.fit(model, train_loader, test_loader)

# Switch to unweighted loss and reset the early stopping callback's state
model.use_weighted = True
early_stop_callback = pl.callbacks.EarlyStopping(monitor="validation_combined_loss", min_delta=0.001, patience=10, verbose=True, mode="min")
logger2 = TensorBoardLogger('logs_visium', name = f"round2_{lr}_{hid_dim}_{head}")

# Train with unweighted loss
trainer2 = pl.Trainer(max_epochs=1000, devices=1, accelerator = "cuda", logger=logger2, callbacks = [early_stop_callback],log_every_n_steps=60)
trainer2.fit(model, train_loader, test_loader)

model.eval()

# %%
model = model.to(device)
out, w = model()

clone_res,ct_res= utils.get_results(out, data, node_encoder_rev, node_encoder_ct)

#%%
path = "data/raw/visium/"
visium = sc.read_visium(path, genome=None, count_file='CytAssist_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5',
                        library_id=None, load_images=True, source_image_path=None)
visium.var_names_make_unique()
coor_int = [[int(x[0]),int(x[1])] for x in visium.obsm["spatial"]]
visium.obsm["spatial"] = np.array(coor_int)
#%%
clones_columns = clone_res.columns
ct_columns = ct_res.columns
visium.obs = visium.obs.join(clone_res).join(ct_res)
visium.obs["clone"] = visium.obs[clones_columns].idxmax(axis = 1)
visium.obs.clone = visium.obs.clone.astype("str")
visium.obs["cell_type"] = visium.obs[ct_columns].idxmax(axis = 1)
#%%
with mpl.rc_context({'axes.facecolor':  'black',
                     'figure.figsize': [10, 10]}):
    sc.pl.spatial(visium, cmap='magma',
                  # show first 8 cell types
                  color=["cell_type", 'clone'],
                img_key='lowres', alpha_img = 0.5,
                 )
#%%
with mpl.rc_context({'axes.facecolor':  'black',
                     'figure.figsize': [10, 10],
                      "font.size" : 25}):
    sc.pl.spatial(visium, cmap='magma',
                  # show first 8 cell types
                  color=clones_columns,
                img_key='lowres', alpha_img = 0.5,
                 )
#%%
annotation = pd.read_excel("data/raw/Requested_Cell_Barcode_Type_Matrices.xlsx.1", sheet_name="Visium", index_col=0)
annotation.Annotation.replace("DCIS #2", "DCIS 1", inplace=True)
annotation.Annotation.replace("DCIS #1", "DCIS 2", inplace=True)
#%%


visium.obs = visium.obs.join(annotation)
#%%
cell_type_colors = {
    # Immune-related categories
    "B Cells": "#2ca02c",
    "CD4+ T Cells": "#2ca02c",
    "CD8+ T Cells": "#2ca02c",
    "IRF7+ DCs": "#2ca02c",
    "LAMP3+ DCs": "#2ca02c",
    "immune": "#2ca02c",
    
    # DCIS categories
    "DCIS 1": "#1f77b4",
    "DCIS 2": "#ff7f0e",
    
    # Tumor and invasive categories
    "Invasive Tumor": "#7f7f7f",
    "Prolif Invasive Tumor": "#7f7f7f",
    "invasive": "#7f7f7f",
    "mixed/invasive": "#7f7f7f",
    
    # Stromal-related categories
    "Endothelial": "#8c564b",
    "Perivascular-Like": "#8c564b",
    "Stromal": "#8c564b",
    "stromal": "#8c564b",
    "stromal/endothelial": "#8c564b",
    "myoepithelial/stromal/immune": "#8c564b",
    "stromal/endothelial/immune": "#8c564b",
    
    # Other categories
    "Macrophages 2": "#17becf",
    "Mast Cells": "#9edae5",
    "Myoepi ACTA2+": "#aec7e8",
    "Myoepi KRT15+": "#ffbb78",
    "Stromal & T Cell Hybrid": "#f7b6d2",
    "T Cell & Tumor Hybrid": "#dbdb8d",
    "unknown": "#c7c7c7",
    "adipocytes": "#fdae6b",
    "mixed": "#fdedae"
}

# You can then access the colors using cell_type_colors[cell_type]

sc.pl.spatial(visium, cmap='magma',
                  # show first 8 cell types
                  color=["cell_type", 'Annotation'],
                img_key='lowres', alpha_img = 0.5,
                wspace = 0.3, palette = cell_type_colors
                 )
plt.tight_layout()
plt.show()

# %%

coordinates = visium.obs[["array_row","array_col"]]
coordinates = coordinates.astype(int)

    
        
# %%
full_df = utils.get_attention_visium(w,node_encoder_rev, data,coordinates)
# %%
visium.obs = visium.obs.join(full_df)
# %%
with mpl.rc_context({'axes.facecolor':  'black',
                     'figure.figsize': [10, 10]}):
    sc.pl.spatial(visium, cmap='magma',
                  # show first 8 cell types
                  color=full_df.columns,
                img_key='lowres', alpha_img = 0.5,
                 )
# %%
visium.obs.columns = [str(x) for x in visium.obs.columns]
visium.write("data/interim/visium_annotated.h5ad")


# %%
