# %%
import torch
import pandas as pd
import matplotlib as mpl

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle   
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger

from torch.optim import Adam
from sklearn.model_selection import train_test_split
import seaborn as sns
from tqdm import tqdm
import scanpy as sc

from torch.optim.lr_scheduler import ReduceLROnPlateau
import spaceTree.utils as utils
import spaceTree.plotting as plotting

from spaceTree.models import *
import os
import matplotlib as mpl
from torch_geometric.loader import NeighborLoader

os.chdir("/home/o313a/clonal_GNN/")

def compute_class_weights(y_train):
    """Calculate class weights based on the class sample count."""
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    return 1. / class_sample_count

norm_sim = np.load("data/interim/clone_dist_over.npy")
norm_sim = torch.tensor(norm_sim)
# %%
edge_types = ["5g"]
i = 0
data = torch.load(f"data/processed/data_xen_{edge_types[i]}.pt")
with open(f'data/processed/full_encoding_xen_{edge_types[i]}.pkl', 'rb') as handle:
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

# Calculate weights for 'y_type'
y_train_type = data.y_type[data.train_mask]
weight_type_values = compute_class_weights(y_train_type)
weight_type = torch.tensor(weight_type_values, dtype=torch.float)

# Calculate weights for 'y_clone'
y_train_clone = data.y_clone[data.train_mask]
weight_clone_values = compute_class_weights(y_train_clone)
weight_clone = torch.tensor(weight_clone_values, dtype=torch.float)

data.num_classes_clone = len(data.y_clone.unique())
data.num_classes_type = len(data.y_type.unique())

#%%
del data.edge_type

train_loader = NeighborLoader(
    data,
    num_neighbors=[10] * 3,
    batch_size=128,input_nodes = data.train_mask
)

valid_loader = NeighborLoader(
    data,
    num_neighbors=[10] * 3,
    batch_size=128,input_nodes = data.test_mask
)
hold_out_loader = NeighborLoader(data,
    num_neighbors=[10] * 3,
    batch_size=128,input_nodes = data.hold_out
)
# %%
device = torch.device('cuda:0')
data = data.to(device)
weight_clone = weight_clone.to(device)
weight_type = weight_type.to(device)
norm_sim = norm_sim.to(device)

# model = GATLightningModule_sampler(data, weight_clone, 
#     weight_type, norm_sim = norm_sim, 
#     learning_rate=0.005, heads=3, dim_h = 120).to('cuda:0')
model = GATLightningModule_sampler(data, weight_clone, 
    weight_type, norm_sim = norm_sim, 
    learning_rate=0.005, heads=3, dim_h = 120).to('cuda:0')

# %%
# logger1 = TensorBoardLogger('xen_multisample', name = "round1")  
early_stop_callback = pl.callbacks.EarlyStopping(monitor="validation_combined_loss", min_delta=0.001, patience=10, verbose=True, mode="min")
trainer1 = pl.Trainer(max_epochs=1000, devices=1, accelerator = "cuda", 
                      callbacks = [early_stop_callback], log_every_n_steps=10)
trainer1.fit(model, train_loader, valid_loader)

# Switch to unweighted loss and reset the early stopping callback's state
model.use_weighted = True
early_stop_callback = pl.callbacks.EarlyStopping(monitor="validation_combined_loss", min_delta=0.001, patience=10, verbose=True, mode="min")
# logger2 = TensorBoardLogger('xen_multisample', name = "round2")
# Train with unweighted loss
trainer2 = pl.Trainer(max_epochs=1000, devices=1, 
                      accelerator = "cuda", callbacks = [early_stop_callback],
                      log_every_n_steps=60)
trainer2.fit(model, train_loader, valid_loader)



model.eval()

model = model.to(device)
# %%
with torch.no_grad():
    out,w = model(data)
# %%
batch_size = 128
model_output_size = data.num_classes_clone + data.num_classes_type - 2
all_predictions = torch.zeros(data.x.size(0), model_output_size, device=device)
# Use a tensor to track which nodes have been seen
seen_nodes = torch.full((data.x.size(0),), False, dtype=torch.bool, device=device)
# Iterate over the NeighborLoader for the hold-out set
model.to(device).eval()
w1 = []
w2 = []
with torch.no_grad():
    for batch in hold_out_loader:
        # Move batch to device
        batch = batch.to(device)
        
        # Forward pass
        out, w = model(batch)
        w1.append(w[0][:batch.num_nodes])
        w2.append(w[1][:batch.num_nodes])
        
        # Get the original nodes in the current batch
        batch_node_indices = batch.n_id[:batch.num_nodes]

        # Update the predictions and the seen nodes
        all_predictions[batch_node_indices] = out
        seen_nodes[batch_node_indices] = True

assert seen_nodes[data.hold_out].all(), "Not all hold_out nodes were seen in the loader"
#%%
w = (torch.cat(w1,1),torch.cat(w2,0))
# clone_res,ct_res= utils.get_results(out, data, node_encoder_rev, node_encoder_ct,node_encoder_clone, activation = "softmax")
clone_res,ct_res= utils.get_results(all_predictions, data, node_encoder_rev, node_encoder_ct,node_encoder_clone, activation = "softmax")

#%%
xenium = sc.read_h5ad("data/interim/scanvi_xenium.h5ad")

#%%
clone_res.index = [str(x) for x in clone_res.index]
ct_res.index = [str(x) for x in ct_res.index]
clones_columns = clone_res.columns
ct_columns = ct_res.columns
xenium.obs = xenium.obs.join(clone_res).join(ct_res)
xenium.obs["clone"] = xenium.obs[clones_columns].idxmax(axis = 1)
xenium.obs.clone = xenium.obs.clone.astype("str")
xenium.obs["cell_type"] = xenium.obs[ct_columns].idxmax(axis = 1)


# %%
plotting.plot_xenium(xenium.obs.x_centroid, xenium.obs.y_centroid, 
            xenium.obs.clone, palette = "tab10")

#%%
ct_palette = sns.color_palette("tab20", len(set(xenium.obs.Annotation)))
ct_palette = dict(zip(set(xenium.obs.Annotation), ct_palette))
#%%
plotting.plot_xenium(xenium.obs.x_centroid, xenium.obs.y_centroid, 
            xenium.obs.Annotation, palette = ct_palette)
plotting.plot_xenium(xenium.obs.x_centroid, xenium.obs.y_centroid, 
            xenium.obs.cell_type, palette = ct_palette)
#%%
_,f1 = plotting.confusion(xenium,"Annotation","cell_type")
#%%
# _,f1 = plotting.confusion(xenium,"Annotation","C_scANVI_cell_type")
#%%
coordinates = xenium.obs[["x_centroid","y_centroid"]]
coordinates = coordinates.astype(int)
#%%
full_df = utils.get_attention(w,node_encoder_rev, data,coordinates)

# # %%
full_df.index = [str(x) for x in full_df.index]
full_df = full_df.fillna(0)
xenium.obs = xenium.obs.join(full_df)
#%%

# %%
for col in full_df.columns:
    print(col)
    plotting.plot_xenium(xenium.obs.x_centroid, xenium.obs.y_centroid,
                xenium.obs[col])

# %%
# %%
xenium.obs.columns = [str(x) for x in xenium.obs.columns]
xenium.write("data/interim/xenium_annotated_sampler.h5ad")
# %%
