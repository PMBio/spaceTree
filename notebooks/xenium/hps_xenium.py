import torch
import pandas as pd
import matplotlib as mpl
import argparse
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATv2Conv
import matplotlib.pyplot as plt
import pickle   
from torch.nn import Linear, Dropout,BatchNorm1d
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger

from torch.utils.data import Dataset
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import seaborn as sns
from torch_geometric.utils import remove_self_loops
from tqdm import tqdm
import scanpy as sc
import itertools
import gc
from torch import cuda

from plot_spatial import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
import os
def ct_f1_score(pred_cell_type1, data, scanvi_res):
    cells_hold_out =[x.detach().cpu().item() for x in data.hold_out]
    pred_cell_type1 = pred_cell_type1.detach().cpu().numpy()
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
class GAT(torch.nn.Module):
  """Graph Attention Network"""
  def __init__(self, data, heads=8, dim_h = 16):
    super().__init__()
    num_node_features = 2
    dim_out_clone = data.num_classes_clone -1
    dim_out_type = data.num_classes_type -1
    self.data = data

    self.gat1 = GATv2Conv(num_node_features, dim_h, heads=heads, edge_dim=1,dropout = 0.3)
    self.skip = Linear(num_node_features, dim_h * heads)
    self.batchnorm1 = BatchNorm1d(dim_h * heads)
    self.gat2 = GATv2Conv(dim_h * heads, dim_h, heads=1, edge_dim=1)
    # self.batchnorm2 = BatchNorm1d(dim_h)
    # self.gat3 = GATv2Conv(dim_h, dim_h, heads=1, edge_dim=1)
    self.classifier_clone1 = Linear(dim_h, dim_h)
    self.classifier_type1 = Linear(dim_h, dim_h)

    self.classifier_clone2 = Linear(dim_h, dim_out_clone)
    self.classifier_type2 = Linear(dim_h, dim_out_type)



  def forward(self):
    x, edge_index, edge_attr = self.data.x, self.data.edge_index, self.data.edge_attr
    h = self.gat1(x, edge_index, edge_attr = edge_attr)
    h = h + self.skip(x)
    h = self.batchnorm1(h)
    h = F.elu(h)
    h,w = self.gat2(h, edge_index, edge_attr = edge_attr,return_attention_weights = True)
    # h = self.batchnorm2(h)
    # h = F.elu(h)
    # h,w = self.gat3(h, edge_index, edge_attr=edge_attr,return_attention_weights = True)
    h = F.elu(h)
    h_type = self.classifier_type1(h)
    h_clone = self.classifier_clone1(h)
    h_type = self.classifier_type2(h_type)
    h_clone = self.classifier_clone2(h_clone)
    h = torch.cat([h_clone, h_type], dim=1)
    return h,w
class GATLightningModule(pl.LightningModule):
    def __init__(self, data, norm_sim, weight_clone, weight_type,learning_rate=1e-3, heads=3, dim_h = 16):
        super().__init__()
        self.model = GAT(data,heads=heads, dim_h = dim_h)
        self.data = data
        self.lr = learning_rate
        self.use_weighted = False
        self.use_f1 = False
        self.norm_sim = norm_sim
        self.weight_clone = weight_clone
        self.weight_type = weight_type

    def weighted_loss(self, probabilities, norm_sim, target, weight):
        weight = weight.to('cuda:0')
        target = target.to(weight.device)
        probabilities = torch.exp(probabilities) 
        similarity = torch.tensor(norm_sim[target, :]).to('cuda:0')
        level_loss = -torch.log((probabilities * similarity).sum(axis=1))
        level_loss_weighted = level_loss * weight[target]
        reduction = (level_loss_weighted / weight[target].sum()).sum()
        return reduction

    def forward(self, data):
        return self.model()

    def training_step(self, batch, batch_idx):
        pred,_ = self.model()
        pred_clone = pred[:, :self.data.num_classes_clone-1].to('cuda:0')
        pred_cell_type = pred[:, self.data.num_classes_clone-1:].to('cuda:0')
        pred_clone = F.log_softmax(pred_clone, dim=1)
        pred_cell_type = F.log_softmax(pred_cell_type, dim=1)
        train_mask = self.data.train_mask.to('cuda:0')


        if self.use_weighted:
            loss_clone = self.weighted_loss(pred_clone[train_mask], self.norm_sim,
                                        self.data.y_clone[train_mask], self.weight_clone)
        else:
            loss_clone = F.nll_loss(pred_clone[train_mask], self.data.y_clone[train_mask].to('cuda:0'), weight = self.weight_clone.to('cuda:0'))
        loss_type = F.nll_loss(pred_cell_type[train_mask], 
                               self.data.y_type[train_mask].to('cuda:0'), weight=self.weight_type.to('cuda:0'))

        loss = torch.sqrt(loss_clone * loss_type)
        
        self.log('train_loss_clone', loss_clone, on_epoch=True, logger=True,on_step=False)
        self.log('train_loss_type', loss_type, on_epoch=True, logger=True,on_step=False)
        self.log('train_combined_loss', loss, on_epoch=True, logger=True, prog_bar=True,on_step=False)

        return loss


    def validation_step(self, batch, batch_idx):
        pred, _ = self.model()
        pred_clone = pred[:, :self.data.num_classes_clone-1].to('cuda:0')
        pred_cell_type = pred[:, self.data.num_classes_clone-1:].to('cuda:0')
        pred_clone = F.log_softmax(pred_clone, dim=1)
        pred_cell_type = F.log_softmax(pred_cell_type, dim=1)
        test_mask = self.data.test_mask.to('cuda:0')
        weight_tensor = self.weight_type.to('cuda:0') 
        weight_tensor_clone = self.weight_clone.to('cuda:0')



        if self.use_weighted:
            loss_clone = self.weighted_loss(pred_clone[self.data.test_mask], self.norm_sim,
                                        self.data.y_clone[self.data.test_mask], weight_tensor_clone)
        else:
            loss_clone = F.nll_loss(pred_clone[test_mask], self.data.y_clone[test_mask], weight=weight_tensor_clone)
        loss_type = F.nll_loss(pred_cell_type[test_mask], 
                               self.data.y_type[test_mask], weight=weight_tensor)

        loss = torch.sqrt(loss_clone * loss_type)

        pred_clone1 = pred_clone.argmax(dim=1)
        pred_cell_type1 = pred_cell_type.argmax(dim=1)

        correct_clones = (pred_clone1[self.data.test_mask] == self.data.y_clone[self.data.test_mask]).sum()
        correct_types = (pred_cell_type1[self.data.test_mask] == self.data.y_type[self.data.test_mask]).sum()

        acc_clone = int(correct_clones) / len(self.data.test_mask)
        acc_type = int(correct_types) / len(self.data.test_mask)
        
        self.log('validation_loss_clone', loss_clone, on_epoch=True, logger=True,on_step=False)
        self.log('validation_loss_type', loss_type, on_epoch=True, logger=True,on_step=False)
        self.log('validation_combined_loss', loss, on_epoch=True, logger=True, prog_bar=True,on_step=False)
        self.log('validation_acc_clone', acc_clone, on_epoch=True, logger=True, prog_bar=True,on_step=False)
        self.log('validation_acc_ct', acc_type, on_epoch=True, logger=True, prog_bar=True,on_step=False)
        if self.use_f1:
            self.log('scanvi_f1', ct_f1_score(pred_cell_type1), on_epoch=True, logger=True, prog_bar=True,on_step=False)

        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "validation_combined_loss",
            "frequency": 1
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
        },
    }


def train_model(d, lr, hd, h):
    xenium = sc.read_h5ad("../data/interim/scanvi_xenium.h5ad")
    scanvi_res = xenium.obs[["C_scANVI_clone","C_scANVI_cell_type"]]
    norm_sim = np.load("../data/interim/clone_dist_over.npy")
    norm_sim = torch.tensor(norm_sim).to('cuda:0')
    scanvi_res = xenium.obs[["C_scANVI_clone","C_scANVI_cell_type"]]

    data = torch.load(f"../data/processed/data_xen{d}.pt")
    with open(f'../data/processed/full_encoding_xen{d}.pkl', 'rb') as handle:
        encoder_dict = pickle.load(handle)
    data.edge_attr = data.edge_attr.reshape((-1,1))
    scanvi_res["C_scANVI_clone"]= scanvi_res["C_scANVI_clone"].map(encoder_dict["clones"])
    scanvi_res["C_scANVI_cell_type"] = scanvi_res["C_scANVI_cell_type"].map(encoder_dict["types"])
    scanvi_res = scanvi_res.reset_index()
    scanvi_res.columns = ["cell_id","C_scANVI_clone","C_scANVI_cell_type"]
    scanvi_res = scanvi_res.dropna(subset = "C_scANVI_cell_type")
    # Separate training data (scRNA) from spatial data
    hold_out_indices = np.where(data.y_clone == -1)[0]
    hold_out = torch.tensor(hold_out_indices, dtype=torch.long)

    total_size = data.x.shape[0] - len(hold_out)
    train_size = int(0.8 * total_size)

    # Get indices that are not in hold_out
    hold_in_indices = np.arange(data.x.shape[0])
    hold_in = [index for index in hold_in_indices if index not in hold_out]
    scanvi_res["in_graph"] = scanvi_res.index.isin(hold_out_indices)
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


    data = data.to('cuda:0')
    model = GATLightningModule(data, norm_sim, weight_clone, weight_type, lr, h, hd).to('cuda:0')
    norm_sim = norm_sim.to('cuda:0')
    weight_clone = weight_clone.to('cuda:0')
    weight_type = weight_type.to('cuda:0')

    logger1 = TensorBoardLogger('xen_hp', name = f"round1_{d}_{lr}_{hd}_{h}")  
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="validation_combined_loss", min_delta=0.001, patience=10, verbose=True, mode="min")
    trainer1 = pl.Trainer(max_epochs=1000, devices=1, logger = logger1,accelerator = "cuda", 
                          callbacks = [early_stop_callback], enable_progress_bar=False)
    trainer1.fit(model, train_loader, test_loader)
    model.use_weighted = True
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="validation_combined_loss", min_delta=0.001, patience=10, verbose=True, mode="min")
    logger2 = TensorBoardLogger('xen_hp', name = f"round2_{d}_{lr}_{hd}_{h}")
    # Train with unweighted loss
    trainer2 = pl.Trainer(max_epochs=1000, devices=1, logger = logger2,
                           accelerator = "cuda", callbacks = [early_stop_callback],enable_progress_bar=False)
    trainer2.fit(model, train_loader, test_loader)
    data = data.to('cuda:0')
    model = model.to('cuda:0')
    pred, _ = model(data)
    pred_cell_type = pred[:, data.num_classes_clone-1:]
    pred_cell_type = F.log_softmax(pred_cell_type, dim=1)
    pred_cell_type1 = pred_cell_type.argmax(dim=1)
    f1 = ct_f1_score(pred_cell_type1, data, scanvi_res)
    res = trainer1.logged_metrics
    res["f1"] = f1
    print("-------------------------------------------")
    print (res["f1"])
    print("-------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model with given hyperparameters.')
    parser.add_argument('--distance', type=int, required=True, help='Distance for the model')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate for the model')
    parser.add_argument('--hid_dim', type=int, required=True, help='Hidden Dimension for the model')
    parser.add_argument('--head', type=int, required=True, help='Head for the model')

    args = parser.parse_args()
    train_model(args.distance, args.lr, args.hid_dim, args.head)

