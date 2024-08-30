import torch

import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.nn import Linear,BatchNorm1d
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau



class GATLightningModule_sampler(pl.LightningModule):
    """
    LightningModule implementation for the GAT (Graph Attention Network) model with sampling.

    Args:
        data_param (object): Object containing the parameters of the input data.
        weight_clone (torch.Tensor): Weight tensor for the clone loss.
        weight_type (torch.Tensor): Weight tensor for the type loss.
        norm_sim (torch.Tensor, optional): Tensor containing the similarity values between clones for tree loss implementation. Defaults to None.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        heads (int, optional): Number of attention heads. Defaults to 3.
        dim_h (int, optional): Hidden dimension size. Defaults to 16.
        weight_decay (float, optional): Weight decay for the optimizer. Defaults to 1e-4.
        map_enteties (str, optional): Mapping entities to predict. Possible values: "both", "clone", "type". Defaults to "both".
        n_layers (int, optional): Number of GAT layers. Defaults to 2.
    """

    def __init__(self, data_param, weight_clone, weight_type, norm_sim=None, learning_rate=1e-3, heads=3, dim_h=16,
                 weight_decay=1e-4, map_enteties="both", n_layers=2):
        super().__init__()
        self.model = GAT2(data_param.num_classes_clone, data_param.num_classes_type, heads, dim_h, map_enteties, data_param.num_node_features)
        self.data = data_param
        self.lr = learning_rate
        self.use_weighted = False
        if norm_sim is not None:
            self.norm_sim = norm_sim
        self.weight_clone = weight_clone
        self.weight_type = weight_type
        self.weight_decay = weight_decay
        self.map_enteties = map_enteties

    def weighted_loss(self, probabilities, norm_sim, target, weight):
        """
        Computes the weighted loss.

        Args:
            probabilities (torch.Tensor): Predicted probabilities.
            norm_sim (torch.Tensor): Tensor containing the similarity values for normalization.
            target (torch.Tensor): Target tensor.
            weight (torch.Tensor): Weight tensor.

        Returns:
            torch.Tensor: Weighted loss.
        """
        probabilities = torch.exp(probabilities) 
        similarity = torch.tensor(norm_sim[target, :]).to(self.device)
        level_loss = -torch.log((probabilities * similarity).sum(axis=1))
        if weight is None:
            reduction = level_loss.mean()
        else:
            level_loss_weighted = (level_loss * weight[target]).sum()
            reduction = level_loss_weighted / weight[target].sum()
        return reduction

    def forward(self, batch):
        """
        Forward pass of the model.

        Args:
            batch (torch.Tensor): Input batch.

        Returns:
            torch.Tensor: Model output.
        """
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch (torch.Tensor): Input batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        pred, _, predicted_features = self.model(batch)
        if self.map_enteties == "both":
            mapping = ["clone", "type"]
            pred_clone = pred[:, :self.data.num_classes_clone-1]
            pred_cell_type = pred[:, self.data.num_classes_clone-1:]
        elif self.map_enteties == "clone":
            mapping = ["clone"]
            pred_clone = pred
        elif self.map_enteties == "type":
            mapping = ["type"]
            pred_cell_type = pred
        if "clone" in mapping:
            if self.use_weighted:
                loss_clone = self.weighted_loss(pred_clone, self.norm_sim, batch.y_clone, self.weight_clone)
            else:
                loss_clone = F.nll_loss(pred_clone, batch.y_clone, reduction='none')
                if self.weight_clone is not None:
                    loss_clone = (loss_clone * self.weight_clone[batch.y_clone]).sum()
                    loss_clone = loss_clone / self.weight_clone[batch.y_clone].sum()
            self.log('train_loss_clone', loss_clone, on_epoch=True, logger=True, on_step=False, batch_size=batch.x.size(0))
            loss = loss_clone
        if "type" in mapping:
            loss_type = F.nll_loss(pred_cell_type, batch.y_type)
            loss_type = (loss_type * self.weight_type[batch.y_type]).sum()
            loss_type = loss_type / self.weight_type[batch.y_type].sum()
            self.log('train_loss_type', loss_type, on_epoch=True, logger=True, on_step=False, batch_size=batch.x.size(0))
            loss = loss_type
        if self.map_enteties == "both":
            loss = torch.sqrt(loss_clone * loss_type)
        self.log('train_combined_loss', loss, on_epoch=True, logger=True, prog_bar=True, on_step=False, batch_size=batch.x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch (torch.Tensor): Input batch.
            batch_idx (int): Batch index.
        """
        pred, _, predicted_features = self.model(batch)
        if self.map_enteties == "both":
            mapping = ["clone", "type"]
            pred_clone = pred[:, :self.data.num_classes_clone-1]
            pred_cell_type = pred[:, self.data.num_classes_clone-1:]
        elif self.map_enteties == "clone":
            mapping = ["clone"]
            pred_clone = pred
        elif self.map_enteties == "type":
            mapping = ["type"]
            pred_cell_type = pred
        if "clone" in mapping:
            if self.use_weighted:
                loss_clone = self.weighted_loss(pred_clone, self.norm_sim, batch.y_clone, self.weight_clone)
            else:
                loss_clone = F.nll_loss(pred_clone, batch.y_clone, reduction='none')
                loss_clone = (loss_clone * self.weight_clone[batch.y_clone]).mean()
            pred_clone1 = pred_clone.argmax(dim=1)
            correct_clones = (pred_clone1 == batch.y_clone).sum()
            acc_clone = int(correct_clones) / len(batch.y_clone)
            self.log('validation_loss_clone', loss_clone, on_epoch=True, logger=True, on_step=False, batch_size=batch.x.size(0))
            self.log('validation_acc_clone', acc_clone, on_epoch=True, logger=True, prog_bar=True, on_step=False, batch_size=batch.x.size(0))
            loss = loss_clone
        if "type" in mapping:
            loss_type = F.nll_loss(pred_cell_type, batch.y_type, reduction='none')
            loss_type = (loss_type * self.weight_type[batch.y_type]).mean()
            pred_cell_type1 = pred_cell_type.argmax(dim=1)
            correct_types = (pred_cell_type1 == batch.y_type).sum()
            acc_type = int(correct_types) / len(batch.y_clone)
            self.log('validation_loss_type', loss_type, on_epoch=True, logger=True, on_step=False, batch_size=batch.x.size(0))
            self.log('validation_acc_ct', acc_type, on_epoch=True, logger=True, prog_bar=True, on_step=False, batch_size=batch.x.size(0))
            loss = loss_type
        if self.map_enteties == "both":
            loss = torch.sqrt(loss_clone * loss_type)
        self.log('validation_combined_loss', loss, on_epoch=True, logger=True, prog_bar=True, on_step=False, batch_size=batch.x.size(0))

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            dict: Dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9, nesterov=True)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=False)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "validation_combined_loss",
                "frequency": 1
            },
        }


class GAT2(torch.nn.Module):
    """
    Graph Attention Network (GAT) model.

    Args:
        num_classes_clone (int): Number of clone classes.
        num_classes_type (int): Number of type classes.
        heads (int, optional): Number of attention heads. Defaults to 1.
        dim_h (int, optional): Hidden dimension size. Defaults to 16.
        map_enteties (str, optional): Mapping entities to predict. Possible values: "both", "clone", "type". Defaults to "both".
        num_node_features (int, optional): Number of node features. Defaults to 2.
    """

    def __init__(self, num_classes_clone, num_classes_type, heads=1, dim_h=16, map_enteties="both", num_node_features=2):
        super().__init__()
        dim_out_clone = num_classes_clone - 1
        dim_out_type = num_classes_type - 1
        self.gat1 = GATv2Conv(num_node_features, dim_h, heads=heads, edge_dim=1, dropout=0.3)
        self.skip = Linear(num_node_features, dim_h * heads)
        self.batchnorm1 = BatchNorm1d(dim_h * heads)
        self.gat2 = GATv2Conv(dim_h * heads, dim_h, heads=1, edge_dim=1)
        self.fc1 = Linear(dim_h, dim_h)
        self.classifier_clone = Linear(dim_h, dim_out_clone)
        self.classifier_type = Linear(dim_h, dim_out_type)
        self.map_enteties = map_enteties

    def forward(self, data):
        """
        Forward pass of the model.

        Args:
            data (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Model output.
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.gat1(x, edge_index, edge_attr=edge_attr)
        h = h + self.skip(x)
        h = self.batchnorm1(h)
        h = F.elu(h)
        h, w = self.gat2(h, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        h = F.elu(h)
        h_last = F.relu(self.fc1(h))
        if self.map_enteties == "both":
            h_type = F.log_softmax(self.classifier_type(h_last), dim=1)
            h_clone = F.log_softmax(self.classifier_clone(h_last), dim=1)
            h = torch.cat([h_clone, h_type], dim=1)
        elif self.map_enteties == "clone":
            h = F.log_softmax(self.classifier_clone(h_last), dim=1)
        elif self.map_enteties == "type":
            h = F.log_softmax(self.classifier_type(h_last), dim=1)
        predicted_features = None
        return h, w, predicted_features

    def get_fc1_embeddings(self, data):
        """
        Get the embeddings from the first fully connected layer.

        Args:
            data (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Embeddings from the first fully connected layer.
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.gat1(x, edge_index, edge_attr=edge_attr)
        h = h + self.skip(x)
        h = self.batchnorm1(h)
        h = F.elu(h)
        h, _ = self.gat2(h, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        h = F.elu(h)
        h = self.fc1(h)
        return h


