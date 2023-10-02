import torch

import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATv2Conv
from torch.nn import Linear,BatchNorm1d
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GAT(torch.nn.Module):
  """Graph Attention Network"""
  def __init__(self, num_classes_clone, num_classes_type, heads=1, dim_h = 16):
    super().__init__()
    num_node_features = 2
    dim_out_clone = num_classes_clone -1
    dim_out_type = num_classes_type -1

    self.gat1 = GATv2Conv(num_node_features, dim_h, heads=heads, edge_dim=1,dropout = 0.3)
    self.skip = Linear(num_node_features, dim_h * heads)
    self.batchnorm1 = BatchNorm1d(dim_h * heads)
    self.gat2 = GATv2Conv(dim_h * heads, dim_h, heads=1, edge_dim=1)
    self.classifier_clone1 = Linear(dim_h, dim_h)
    self.classifier_type1 = Linear(dim_h, dim_h)

    self.classifier_clone2 = Linear(dim_h, dim_out_clone)
    self.classifier_type2 = Linear(dim_h, dim_out_type)



  def forward(self,data):
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    h = self.gat1(x, edge_index, edge_attr = edge_attr)
    h = h + self.skip(x)
    h = self.batchnorm1(h)
    h = F.elu(h)
    h,w = self.gat2(h, edge_index, edge_attr = edge_attr,return_attention_weights = True)
    h = F.elu(h)
    h_type = self.classifier_type1(h)
    h_clone = self.classifier_clone1(h)
    h_type = self.classifier_type2(h_type)
    h_clone = self.classifier_clone2(h_clone)
    h = torch.cat([h_clone, h_type], dim=1)
    return h,w
  

class GCN(torch.nn.Module):
    def __init__(self, dim_h, num_node_features,num_classes_clone, num_classes_type):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)
        # self.conv_clone = GCNConv(16, data.num_classes_clone - 1)
        # self.conv_type = GCNConv(16, data.num_classes_type-1)
        self.out = GCNConv(dim_h,num_classes_clone +  num_classes_type -2)


    def forward(self,data):
        x = self.conv1(data.x, data.edge_index, data.edge_weight)
        x = self.conv2(x, data.edge_index, data.edge_weight)
        x = self.conv3(x, data.edge_index, data.edge_weight)
        x_pre = F.relu(x)
        x_out = self.out(x_pre, data.edge_index, data.edge_weight)
        return (x_out, 0)
  
class GATLightningModule(pl.LightningModule):
    def __init__(self, data, weight_clone, weight_type, norm_sim = None, learning_rate=1e-3, heads=3, dim_h = 16, model_type = 'GAT'):
        super().__init__()
        if model_type == 'GAT':
            self.model = GAT(data.num_classes_clone, data.num_classes_type, heads, dim_h)
        elif model_type == 'GCN':
            self.model = GCN(dim_h, data.num_node_features,data.num_classes_clone, data.num_classes_type)
        else:
            raise ValueError('Model type not recognized')
        self.data = data
        self.lr = learning_rate
        self.use_weighted = False
        if norm_sim is not None:
            self.norm_sim = norm_sim
        self.weight_clone = weight_clone
        self.weight_type = weight_type

    def weighted_loss(self, probabilities, norm_sim, target, weight):
        probabilities = torch.exp(probabilities) 
        similarity = torch.tensor(norm_sim[target, :]).to(self.device)
        level_loss = -torch.log((probabilities * similarity).sum(axis=1))
        level_loss_weighted = level_loss * weight[target]
        reduction = (level_loss_weighted / weight[target].sum()).sum()
        return reduction

    def forward(self,data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        pred,_ = self.model(self.data)
        pred_clone = pred[:, :self.data.num_classes_clone-1]
        pred_cell_type = pred[:, self.data.num_classes_clone-1:]
        pred_clone = F.log_softmax(pred_clone, dim=1)
        pred_cell_type = F.log_softmax(pred_cell_type, dim=1)

        if self.use_weighted:
            loss_clone = self.weighted_loss(pred_clone[self.data.train_mask], self.norm_sim,
                                        self.data.y_clone[self.data.train_mask], self.weight_clone)
        else:
            loss_clone = F.nll_loss(pred_clone[self.data.train_mask], self.data.y_clone[self.data.train_mask], weight = self.weight_clone)
        loss_type = F.nll_loss(pred_cell_type[self.data.train_mask], 
                               self.data.y_type[self.data.train_mask], weight=self.weight_type)

        loss = torch.sqrt(loss_clone * loss_type)
        
        self.log('train_loss_clone', loss_clone, on_epoch=True, logger=True,on_step=False)
        self.log('train_loss_type', loss_type, on_epoch=True, logger=True,on_step=False)
        self.log('train_combined_loss', loss, on_epoch=True, logger=True, prog_bar=True,on_step=False)

        return loss


    def validation_step(self, batch, batch_idx):
        pred, _ = self.model(self.data)
        pred_clone = pred[:, :self.data.num_classes_clone-1]
        pred_cell_type = pred[:, self.data.num_classes_clone-1:]
        pred_clone = F.log_softmax(pred_clone, dim=1)
        pred_cell_type = F.log_softmax(pred_cell_type, dim=1)
        
        if self.use_weighted:
            loss_clone = self.weighted_loss(pred_clone[self.data.test_mask], self.norm_sim,
                                        self.data.y_clone[self.data.test_mask], self.weight_clone)
        else:
            loss_clone = F.nll_loss(pred_clone[self.data.test_mask], self.data.y_clone[self.data.test_mask], weight = self.weight_clone)
        loss_type = F.nll_loss(pred_cell_type[self.data.test_mask], 
                               self.data.y_type[self.data.test_mask], weight=self.weight_type)

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


class GATLightningModule_sampler(pl.LightningModule):
    def __init__(self, data, weight_clone, weight_type, norm_sim = None, learning_rate=1e-3, heads=3, dim_h = 16, model_type = 'GAT'):
        super().__init__()
        if model_type == 'GAT':
            self.model = GAT(data.num_classes_clone, data.num_classes_type, heads, dim_h)
        elif model_type == 'GCN':
            self.model = GCN(dim_h, data.num_node_features,data.num_classes_clone, data.num_classes_type)
        else:
            raise ValueError('Model type not recognized')
        self.data = data
        self.lr = learning_rate
        self.use_weighted = False
        if norm_sim is not None:
            self.norm_sim = norm_sim
        self.weight_clone = weight_clone
        self.weight_type = weight_type

    def weighted_loss(self, probabilities, norm_sim, target, weight):
        probabilities = torch.exp(probabilities) 
        similarity = torch.tensor(norm_sim[target, :]).to(self.device)
        level_loss = -torch.log((probabilities * similarity).sum(axis=1))
        level_loss_weighted = level_loss * weight[target]
        reduction = (level_loss_weighted / weight[target].sum()).sum()
        return reduction

    def forward(self,data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        pred,_ = self.model(batch)
        pred_clone = pred[:, :self.data.num_classes_clone-1]
        pred_cell_type = pred[:, self.data.num_classes_clone-1:]
        pred_clone = F.log_softmax(pred_clone, dim=1)
        pred_cell_type = F.log_softmax(pred_cell_type, dim=1)

        if self.use_weighted:
            loss_clone = self.weighted_loss(pred_clone, self.norm_sim,
                                        batch.y_clone, self.weight_clone)
        else:
            loss_clone = F.nll_loss(pred_clone, batch.y_clone, reduction = 'none')
            loss_clone = (loss_clone*self.weight_clone[batch.y_clone]).mean()
        loss_type = F.nll_loss(pred_cell_type, 
                               batch.y_type)
        loss_type = (loss_type*self.weight_type[batch.y_type]).mean()

        loss = torch.sqrt(loss_clone * loss_type)
        
        self.log('train_loss_clone', loss_clone, on_epoch=True, logger=True,on_step=False)
        self.log('train_loss_type', loss_type, on_epoch=True, logger=True,on_step=False)
        self.log('train_combined_loss', loss, on_epoch=True, logger=True, prog_bar=True,on_step=False)

        return loss


    def validation_step(self, batch, batch_idx):
        pred, _ = self.model(batch)
        pred_clone = pred[:, :self.data.num_classes_clone-1]
        pred_cell_type = pred[:, self.data.num_classes_clone-1:]
        pred_clone = F.log_softmax(pred_clone, dim=1)
        pred_cell_type = F.log_softmax(pred_cell_type, dim=1)
        
        if self.use_weighted:
            loss_clone = self.weighted_loss(pred_clone, self.norm_sim,
                                        batch.y_clone, self.weight_clone)
        else:
            loss_clone = F.nll_loss(pred_clone, batch.y_clone, reduction = 'none')
            loss_clone = (loss_clone*self.weight_clone[batch.y_clone]).mean()
        loss_type = F.nll_loss(pred_cell_type, 
                               batch.y_type, reduction = 'none')
        loss_type = (loss_type*self.weight_type[batch.y_type]).mean()

        loss = torch.sqrt(loss_clone * loss_type)

        pred_clone1 = pred_clone.argmax(dim=1)
        pred_cell_type1 = pred_cell_type.argmax(dim=1)

        correct_clones = (pred_clone1 == batch.y_clone).sum()
        correct_types = (pred_cell_type1 == batch.y_type).sum()

        acc_clone = int(correct_clones) / len(batch)
        acc_type = int(correct_types) / len(batch)
        
        self.log('validation_loss_clone', loss_clone, on_epoch=True, logger=True,on_step=False)
        self.log('validation_loss_type', loss_type, on_epoch=True, logger=True,on_step=False)
        self.log('validation_combined_loss', loss, on_epoch=True, logger=True, prog_bar=True,on_step=False)
        self.log('validation_acc_clone', acc_clone, on_epoch=True, logger=True, prog_bar=True,on_step=False)
        self.log('validation_acc_ct', acc_type, on_epoch=True, logger=True, prog_bar=True,on_step=False)

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