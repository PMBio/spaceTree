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
from sklearn.metrics import f1_score
import spaceTree.utils as utils
from spaceTree.models import *
import os
import matplotlib as mpl
from sklearn.metrics import confusion_matrix
from torch_geometric.loader import NeighborLoader


os.chdir("/home/o313a/clonal_GNN/")

def compute_class_weights(y_train):
    """Calculate class weights based on the class sample count."""
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    return 1. / class_sample_count

norm_sim = np.load("data/interim/clone_dist_over.npy")
norm_sim = torch.tensor(norm_sim)

# %%
d = 20
data = torch.load(f"data/processed/data_xen{d}.pt")
with open(f'data/processed/full_encoding_xen{d}.pkl', 'rb') as handle:
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

train_loader = NeighborLoader(
    data,
    num_neighbors=[10] * 3,
    batch_size=128,input_nodes = data.test_mask
)

# %%
device = torch.device('cuda:0')
data = data.to(device)
weight_clone = weight_clone.to(device)
weight_type = weight_type.to(device)
norm_sim = norm_sim.to(device)
loader = NeighborLoader(
    data,
    num_neighbors=[10] * 3,
    batch_size=128,
)
model = GATLightningModule_sampler(data, weight_clone, weight_type, norm_sim = norm_sim, learning_rate=0.005, heads=3, dim_h = 32).to('cuda:0')

# %%
batch = next(iter(loader))

# %%
pred, _ = model(batch)

# %%
        pred_clone = pred[:, :data.num_classes_clone-1]
        pred_cell_type = pred[:, data.num_classes_clone-1:]
        pred_clone = F.log_softmax(pred_clone, dim=1)
        pred_cell_type = F.log_softmax(pred_cell_type, dim=1)

# %%
loss_clone = F.nll_loss(pred_clone, batch.y_clone, weight = self.weight_clone)

# %%
loss_clone = F.nll_loss(pred_clone, batch.y_clone, weight = weight_clone)

# %%
weight_clone

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
from sklearn.metrics import f1_score
import spaceTree.utils as utils
from spaceTree.models import *
import os
import matplotlib as mpl
from sklearn.metrics import confusion_matrix
from torch_geometric.loader import NeighborLoader


os.chdir("/home/o313a/clonal_GNN/")

def compute_class_weights(y_train):
    """Calculate class weights based on the class sample count."""
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    return 1. / class_sample_count

norm_sim = np.load("data/interim/clone_dist_over.npy")
norm_sim = torch.tensor(norm_sim)

# %%
d = 20
data = torch.load(f"data/processed/data_xen{d}.pt")
with open(f'data/processed/full_encoding_xen{d}.pkl', 'rb') as handle:
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

# %%
weight_clone

#%%
del data.edge_type

train_loader = NeighborLoader(
    data,
    num_neighbors=[10] * 3,
    batch_size=128,input_nodes = data.train_mask
)

train_loader = NeighborLoader(
    data,
    num_neighbors=[10] * 3,
    batch_size=128,input_nodes = data.test_mask
)

# %%
device = torch.device('cuda:0')
data = data.to(device)
weight_clone = weight_clone.to(device)
weight_type = weight_type.to(device)
norm_sim = norm_sim.to(device)
loader = NeighborLoader(
    data,
    num_neighbors=[10] * 3,
    batch_size=128,
)
model = GATLightningModule_sampler(data, weight_clone, weight_type, norm_sim = norm_sim, learning_rate=0.005, heads=3, dim_h = 32).to('cuda:0')

# %%

batch = next(iter(train_loader))

# %%
pred, _ = model(batch)

# %%
        pred_clone = pred[:, :data.num_classes_clone-1]
        pred_cell_type = pred[:, data.num_classes_clone-1:]
        pred_clone = F.log_softmax(pred_clone, dim=1)
        pred_cell_type = F.log_softmax(pred_cell_type, dim=1)

# %%
pred_clone

# %%
F.nll_loss(pred_clone, batch.y_clone)

# %%
pred_clone.shape

# %%
weight_clone

# %%
weight_clone.shape

# %%
batch.y_clone.shape

# %%
batch.y_clone

# %%
batch.y_clone.max() < 18 and batch.y_clone.min() >= 0

# %%
pred_clone_cpu = pred_clone.cpu()
batch_y_clone_cpu = batch.y_clone.cpu()
weight_clone_cpu = weight_clone.cpu()
loss = F.nll_loss(pred_clone_cpu, batch_y_clone_cpu, weight=weight_clone_cpu)

# %%
loss


