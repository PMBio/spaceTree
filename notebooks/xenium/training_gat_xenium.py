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

os.chdir("/home/o313a/clonal_GNN/")

def ct_f1_score(pred_cell_type1):
    cells_hold_out =[x.item() for x in data.hold_out]
    ct_res = pd.DataFrame(pred_cell_type1[data.hold_out.detach().cpu().numpy()], index = cells_hold_out)
    ct_res.columns = ["cell_type"]
    ct_res = ct_res.loc[scanvi_res[scanvi_res.in_graph].index]
    return f1_score(ct_res.cell_type, scanvi_res[scanvi_res.in_graph].C_scANVI_cell_type,average='weighted')
def compute_class_weights(y_train):
    """Calculate class weights based on the class sample count."""
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    return 1. / class_sample_count
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

# Create dummy datasets for training and testing
train_dataset = GraphDataset(data, data.train_mask)
test_dataset = GraphDataset(data, data.test_mask)

# Create DataLoaders
# Note: Since the dataset contains only one graph, shuffling won't have an effect
train_loader = DataLoader(train_dataset, batch_size=1)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# %%
device = torch.device('cuda:0')
data = data.to(device)
weight_clone = weight_clone.to(device)
weight_type = weight_type.to(device)
norm_sim = norm_sim.to(device)
model = GATLightningModule(data, weight_clone, weight_type, norm_sim = norm_sim, learning_rate=0.005, heads=3, dim_h = 32).to('cuda:0')


# %%
# logger1 = TensorBoardLogger('xen_training', name = "weighted_loss_improv")  
early_stop_callback = pl.callbacks.EarlyStopping(monitor="validation_combined_loss", min_delta=0.001, patience=10, verbose=True, mode="min")
# Train
trainer1 = pl.Trainer(max_epochs=1000, devices=1, accelerator = "cuda", callbacks = [early_stop_callback], log_every_n_steps=10)
trainer1.fit(model, train_loader, test_loader)

# %%
# Switch to unweighted loss and reset the early stopping callback's state
model.use_weighted = True
early_stop_callback = pl.callbacks.EarlyStopping(monitor="validation_combined_loss", min_delta=0.001, patience=10, verbose=True, mode="min")
# logger2 = TensorBoardLogger('xen_training', name = "finetuning")
# %%
# Train with unweighted loss
trainer2 = pl.Trainer(max_epochs=1000, devices=1, accelerator = "cuda", callbacks = [early_stop_callback],log_every_n_steps=60)
trainer2.fit(model, train_loader, test_loader)

# %%
model.eval()



# %%
model = model.to(device)
out, w = model(data)

clone_res,ct_res= utils.get_results(out, data, node_encoder_rev, node_encoder_ct)
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
utils.plot_xenium(xenium.obs.x_centroid, xenium.obs.y_centroid, 
            xenium.obs.clone, palette = "tab20")
# %%
utils.plot_xenium(xenium.obs.x_centroid, xenium.obs.y_centroid, 
            xenium.obs.cell_type, palette = "tab20")

# %%
annotation = pd.read_excel("data/raw/Requested_Cell_Barcode_Type_Matrices.xlsx.1", sheet_name="Xenium R1 Fig1-5 (supervised)", index_col=0)
annotation.Cluster.replace("DCIS_2", "DCIS 1", inplace=True)
annotation.Cluster.replace("DCIS_1", "DCIS 2", inplace=True)
annotation.columns = ["Annotation"]
annotation.index = [str(x) for x in annotation.index]
# %%

xenium.obs = xenium.obs.join(annotation)

# %%
#%%
xenium.obs.cell_type.fillna("unknown", inplace = True)
list1 = list(set(xenium.obs.cell_type))
list2 = list(set(xenium.obs.Annotation))
mapping_dict = {}

for name in list2:
    if isinstance(name, str):
        replaced_name = name.replace('_', ' ')
        if replaced_name in list1:
            mapping_dict[name] = replaced_name
        elif name == "Unlabeled":
            mapping_dict[name] = 'unknown'
            mapping_dict['Unlabeled'] = 'unknown'
        else:
            mapping_dict[name] = name
    else:
        mapping_dict[name] = 'unknown'
mapping_dict['DCIS 1'] = 'DCIS 2'
mapping_dict['DCIS 2'] = 'DCIS 1'

#%%
xenium.obs.Annotation = xenium.obs.Annotation.map(mapping_dict)
#%%
utils.plot_xenium(xenium.obs.x_centroid, xenium.obs.y_centroid, 
            xenium.obs.Annotation, palette = "tab20")

#%%
labels = list(set(xenium.obs.Annotation).intersection(set(xenium.obs.cell_type)))
c_mat = pd.DataFrame(confusion_matrix(xenium.obs.Annotation, xenium.obs.cell_type, labels = labels, normalize = 'true'), index = labels, columns = labels)
#%%
sns.heatmap(c_mat, linewidth=.5, cmap = "Blues", annot = True, fmt=".1f")

# %%
def f1_from_confusion(c_mat):
    # Assuming c_mat is the confusion matrix
    # For a normalized confusion matrix, diagonal elements represent recall for each class
    recall = np.diag(c_mat)
    
    # Precision for each class can be calculated as the diagonal element divided by the sum of elements in the column
    precision = np.diag(c_mat) / np.sum(c_mat, axis=0)
    
    # Calculate the F1 score for each class
    f1 = 2 * precision * recall / (precision + recall)
    
    # If you want the average F1 score, take the mean of the F1 scores for each class
    avg_f1 = np.nanmean(f1)
    
    return f1, avg_f1
f1, avg_f1 = f1_from_confusion(c_mat)
print(avg_f1)
#0.6273974154825895
#%%
coordinates = xenium.obs[["x_centroid","y_centroid"]]
coordinates = coordinates.astype(int)
#%%
full_df = utils.get_attention(w,node_encoder_rev, data,coordinates)

# %%
full_df.index = [str(x) for x in full_df.index]
full_df = full_df.fillna(0)
xenium.obs = xenium.obs.join(full_df)
#%%

# %%
for col in full_df.columns:
    print(col)
    utils.plot_xenium(xenium.obs.x_centroid, xenium.obs.y_centroid,
                xenium.obs[col])

# %%
# %%
xenium.obs.columns = [str(x) for x in xenium.obs.columns]
xenium.write("data/interim/xenium_annotated.h5ad")
# %%
