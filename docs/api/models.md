---
layout: default
title: "spaceTree model"
parent: API Reference
nav_order: 1
---
<!-- markdownlint-disable -->

<a href="../spaceTree/models.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `models`






---

<a href="../spaceTree/models.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GATLightningModule_sampler`
LightningModule implementation for the GAT (Graph Attention Network) model with sampling. 



**Args:**
 
 - <b>`data_param`</b> (object):  Object containing the parameters of the input data. 
 - <b>`weight_clone`</b> (torch.Tensor):  Weight tensor for the clone loss. 
 - <b>`weight_type`</b> (torch.Tensor):  Weight tensor for the type loss. 
 - <b>`norm_sim`</b> (torch.Tensor, optional):  Tensor containing the similarity values between clones for tree loss implementation. Defaults to None. 
 - <b>`learning_rate`</b> (float, optional):  Learning rate for the optimizer. Defaults to 1e-3. 
 - <b>`heads`</b> (int, optional):  Number of attention heads. Defaults to 3. 
 - <b>`dim_h`</b> (int, optional):  Hidden dimension size. Defaults to 16. 
 - <b>`weight_decay`</b> (float, optional):  Weight decay for the optimizer. Defaults to 1e-4. 
 - <b>`map_enteties`</b> (str, optional):  Mapping entities to predict. Possible values: "both", "clone", "type". Defaults to "both". 
 - <b>`n_layers`</b> (int, optional):  Number of GAT layers. Defaults to 2. 

<a href="../spaceTree/models.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GAT2`
Graph Attention Network (GAT) model. 



**Args:**
 
 - <b>`num_classes_clone`</b> (int):  Number of clone classes. 
 - <b>`num_classes_type`</b> (int):  Number of type classes. 
 - <b>`heads`</b> (int, optional):  Number of attention heads. Defaults to 1. 
 - <b>`dim_h`</b> (int, optional):  Hidden dimension size. Defaults to 16. 
 - <b>`map_enteties`</b> (str, optional):  Mapping entities to predict. Possible values: "both", "clone", "type". Defaults to "both". 
 - <b>`num_node_features`</b> (int, optional):  Number of node features. Defaults to 2. 

<a href="../spaceTree/models.py#L196"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>


---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
