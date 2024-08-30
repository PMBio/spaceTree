---
layout: default
title: "Models API"
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

### <kbd>method</kbd> `__init__`

```python
__init__(
    data_param,
    weight_clone,
    weight_type,
    norm_sim=None,
    learning_rate=0.001,
    heads=3,
    dim_h=16,
    weight_decay=0.0001,
    map_enteties='both',
    n_layers=2
)
```






---

#### <kbd>property</kbd> automatic_optimization

If set to ``False`` you are responsible for calling ``.backward()``, ``.step()``, ``.zero_grad()``. 

---

#### <kbd>property</kbd> current_epoch

The current epoch in the ``Trainer``, or 0 if not attached. 

---

#### <kbd>property</kbd> device





---

#### <kbd>property</kbd> dtype





---

#### <kbd>property</kbd> example_input_array

The example input array is a specification of what the module can consume in the :meth:`forward` method. The return type is interpreted as follows: 


-   Single tensor: It is assumed the model takes a single argument, i.e.,  ``model.forward(model.example_input_array)`` 
-   Tuple: The input array should be interpreted as a sequence of positional arguments, i.e.,  ``model.forward(*model.example_input_array)`` 
-   Dict: The input array represents named keyword arguments, i.e.,  ``model.forward(**model.example_input_array)`` 

---

#### <kbd>property</kbd> fabric





---

#### <kbd>property</kbd> global_rank

The index of the current process across all nodes and devices. 

---

#### <kbd>property</kbd> global_step

Total training batches seen across all epochs. 

If no Trainer is attached, this propery is 0. 

---

#### <kbd>property</kbd> hparams

The collection of hyperparameters saved with :meth:`save_hyperparameters`. It is mutable by the user. For the frozen set of initial hyperparameters, use :attr:`hparams_initial`. 



**Returns:**
  Mutable hyperparameters dictionary 

---

#### <kbd>property</kbd> hparams_initial

The collection of hyperparameters saved with :meth:`save_hyperparameters`. These contents are read-only. Manual updates to the saved hyperparameters can instead be performed through :attr:`hparams`. 



**Returns:**
 
 - <b>`AttributeDict`</b>:  immutable initial hyperparameters 

---

#### <kbd>property</kbd> local_rank

The index of the current process within a single node. 

---

#### <kbd>property</kbd> logger

Reference to the logger object in the Trainer. 

---

#### <kbd>property</kbd> loggers

Reference to the list of loggers in the Trainer. 

---

#### <kbd>property</kbd> on_gpu

Returns ``True`` if this model is currently located on a GPU. 

Useful to set flags around the LightningModule for different CPU vs GPU behavior. 

---

#### <kbd>property</kbd> trainer







---

<a href="../spaceTree/models.py#L164"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `configure_optimizers`

```python
configure_optimizers()
```

Configure the optimizer and learning rate scheduler. 



**Returns:**
 
 - <b>`dict`</b>:  Dictionary containing the optimizer and learning rate scheduler. 

---

<a href="../spaceTree/models.py#L65"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(batch)
```

Forward pass of the model. 



**Args:**
 
 - <b>`batch`</b> (torch.Tensor):  Input batch. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Model output. 

---

<a href="../spaceTree/models.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `training_step`

```python
training_step(batch, batch_idx)
```

Training step. 



**Args:**
 
 - <b>`batch`</b> (torch.Tensor):  Input batch. 
 - <b>`batch_idx`</b> (int):  Batch index. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Loss value. 

---

<a href="../spaceTree/models.py#L120"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `validation_step`

```python
validation_step(batch, batch_idx)
```

Validation step. 



**Args:**
 
 - <b>`batch`</b> (torch.Tensor):  Input batch. 
 - <b>`batch_idx`</b> (int):  Batch index. 

---

<a href="../spaceTree/models.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `weighted_loss`

```python
weighted_loss(probabilities, norm_sim, target, weight)
```

Computes the weighted loss. 



**Args:**
 
 - <b>`probabilities`</b> (torch.Tensor):  Predicted probabilities. 
 - <b>`norm_sim`</b> (torch.Tensor):  Tensor containing the similarity values for normalization. 
 - <b>`target`</b> (torch.Tensor):  Target tensor. 
 - <b>`weight`</b> (torch.Tensor):  Weight tensor. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Weighted loss. 


---

<a href="../spaceTree/models.py#L183"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

### <kbd>method</kbd> `__init__`

```python
__init__(
    num_classes_clone,
    num_classes_type,
    heads=1,
    dim_h=16,
    map_enteties='both',
    num_node_features=2
)
```








---

<a href="../spaceTree/models.py#L209"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(data)
```

Forward pass of the model. 



**Args:**
 
 - <b>`data`</b> (torch.Tensor):  Input data. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Model output. 

---

<a href="../spaceTree/models.py#L238"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_fc1_embeddings`

```python
get_fc1_embeddings(data)
```

Get the embeddings from the first fully connected layer. 



**Args:**
 
 - <b>`data`</b> (torch.Tensor):  Input data. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Embeddings from the first fully connected layer. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
