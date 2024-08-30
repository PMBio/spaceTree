---
layout: default
title: "utils API"
parent: API Reference
nav_order: 4
---
<!-- markdownlint-disable -->

<a href="../spaceTree/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `utils`





---

<a href="../spaceTree/utils.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `reverse_log_softmax`

```python
reverse_log_softmax(log_probs)
```

Reverse the log softmax operation to obtain the logits. 



**Args:**
 
 - <b>`log_probs`</b> (torch.Tensor):  The log probabilities. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  The logits. 


---

<a href="../spaceTree/utils.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_results`

```python
get_results(
    pred,
    data,
    node_encoder_rev,
    node_encoder_ct,
    node_encoder_cl,
    activation=None
)
```

Get the results of the prediction for clone and cell type classifications. 



**Args:**
 
 - <b>`pred`</b> (torch.Tensor):  The prediction tensor. 
 - <b>`data`</b> (torch.Tensor):  The data tensor. 
 - <b>`node_encoder_rev`</b> (dict):  The reverse node encoder dictionary. 
 - <b>`node_encoder_ct`</b> (dict):  The cell type node encoder dictionary. 
 - <b>`node_encoder_cl`</b> (dict):  The clone node encoder dictionary. 
 - <b>`activation`</b> (str, optional):  The activation function to apply. Defaults to None. 



**Returns:**
 
 - <b>`tuple`</b>:  A tuple containing the clone results and cell type results as pandas DataFrames. 


---

<a href="../spaceTree/utils.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_results_all`

```python
get_results_all(
    pred,
    data,
    node_encoder_rev,
    node_encoder_ct,
    node_encoder_cl,
    activation=None
)
```

Get the results for clone and cell type predictions. 



**Args:**
 
 - <b>`pred`</b> (torch.Tensor):  Predictions tensor. 
 - <b>`data`</b> (torch_geometric.data.Data):  Input data. 
 - <b>`node_encoder_rev`</b> (dict):  Reverse node encoder dictionary. 
 - <b>`node_encoder_ct`</b> (dict):  Cell type node encoder dictionary. 
 - <b>`node_encoder_cl`</b> (dict):  Clone node encoder dictionary. 
 - <b>`activation`</b> (str, optional):  Activation function to apply.   Can be "softmax", "raw", or None. Defaults to None. 



**Returns:**
 
 - <b>`tuple`</b>:  A tuple containing two pandas DataFrames: 
        - clone_res: DataFrame containing clone predictions. 
        - ct_res: DataFrame containing cell type predictions. 


---

<a href="../spaceTree/utils.py#L102"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_calibrated_results`

```python
get_calibrated_results(
    pred,
    data,
    node_encoder_rev,
    node_encoder_ct,
    node_encoder_cl,
    t
)
```

Calibrates the predicted results using temperature scaling. 



**Args:**
 
 - <b>`pred`</b> (numpy.ndarray):  The predicted results. 
 - <b>`data`</b> (pandas.DataFrame):  The input data. 
 - <b>`node_encoder_rev`</b> (dict):  The reverse node encoder dictionary. 
 - <b>`node_encoder_ct`</b> (dict):  The cell type node encoder dictionary. 
 - <b>`node_encoder_cl`</b> (dict):  The clone node encoder dictionary. 
 - <b>`t`</b> (float):  The temperature parameter for scaling. 



**Returns:**
 
 - <b>`tuple`</b>:  A tuple containing the calibrated results for clones and cell types. 


---

<a href="../spaceTree/utils.py#L131"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_results_clone`

```python
get_results_clone(
    pred,
    data,
    node_encoder_rev,
    node_encoder_cl,
    activation=None
)
```

Get the clone results based on the predictions. 



**Args:**
 
 - <b>`pred`</b> (torch.Tensor):  The predictions. 
 - <b>`data`</b> (Data):  The data object containing the hold_out indices. 
 - <b>`node_encoder_rev`</b> (dict):  The reverse node encoder dictionary. 
 - <b>`node_encoder_cl`</b> (dict):  The node encoder dictionary. 
 - <b>`activation`</b> (str, optional):  The activation function to apply. Defaults to None. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  The clone results. 


---

<a href="../spaceTree/utils.py#L157"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_results_type`

```python
get_results_type(pred, data, node_encoder_rev, node_encoder_ct, activation=None)
```

Get the results type for the predicted cell types. 



**Args:**
 
 - <b>`pred`</b> (torch.Tensor):  The predicted cell types. 
 - <b>`data`</b> (torch.Tensor):  The input data. 
 - <b>`node_encoder_rev`</b> (dict):  A dictionary mapping node indices to cell names. 
 - <b>`node_encoder_ct`</b> (dict):  A dictionary mapping node indices to cell types. 
 - <b>`activation`</b> (str, optional):  The activation function to apply. Defaults to None. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  A DataFrame containing the predicted cell types for the hold-out cells. 


---

<a href="../spaceTree/utils.py#L183"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rotate_90_degrees_clockwise`

```python
rotate_90_degrees_clockwise(matrix)
```

Rotates a matrix 90 degrees clockwise. 



**Parameters:**
 matrix (numpy.ndarray): The input matrix to be rotated. 



**Returns:**
 numpy.ndarray: The rotated matrix. 


---

<a href="../spaceTree/utils.py#L213"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_attention_visium`

```python
get_attention_visium(w, node_encoder_rev, data, coordinates)
```

Calculate attention visualization for Visium data. 



**Args:**
 
 - <b>`w`</b> (tuple):  Tuple containing the edges and weights of the attention graph. 
 - <b>`node_encoder_rev`</b> (dict):  Reverse node encoder dictionary. 
 - <b>`data`</b> (torch.Tensor):  Hold out data. 
 - <b>`coordinates`</b> (pd.DataFrame):  DataFrame containing the coordinates of the nodes. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  DataFrame containing the attention weights for each target node and distance category. 


---

<a href="../spaceTree/utils.py#L286"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_attention`

```python
get_attention(w, node_encoder_rev, data, coordinates)
```

Calculate attention weights for spatial graph nodes based on the given inputs. 



**Args:**
 
 - <b>`w`</b> (tuple):  A tuple containing two elements - edges and weight. 
               - edges (torch.Tensor): Tensor representing the edges of the graph. 
               - weight (torch.Tensor): Tensor representing the weights of the edges. 
 - <b>`node_encoder_rev`</b> (dict):  A dictionary mapping node indices to their corresponding IDs. 
 - <b>`data`</b> (torch.Tensor):  Tensor representing the hold-out data. 
 - <b>`coordinates`</b> (pd.DataFrame):  DataFrame containing the coordinates of the nodes. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  DataFrame containing the attention weights for each target node, categorized by distance. 


---

<a href="../spaceTree/utils.py#L356"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_metrics`

```python
plot_metrics(stored_metrics)
```

Plots the validation accuracy for clone and cell type metrics. 



**Args:**
 
 - <b>`stored_metrics`</b> (dict):  A dictionary containing the stored metrics. 



**Returns:**
 None 


---

<a href="../spaceTree/utils.py#L380"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_class_distributions`

```python
check_class_distributions(
    data,
    weight_clone,
    weight_type,
    norm_sim,
    no_diploid=False
)
```

Check the class distributions in the data and validate the inputs. 



**Args:**
 
 - <b>`data`</b> (torch_geometric.data.Data):  The input data. 
 - <b>`weight_clone`</b> (list):  The weights for each clone class. 
 - <b>`weight_type`</b> (list):  The weights for each type class. 
 - <b>`norm_sim`</b> (torch.Tensor):  The similarity scores. 
 - <b>`no_diploid`</b> (bool, optional):  Whether to exclude the diploid class. Defaults to False. 



**Raises:**
 
 - <b>`AssertionError`</b>:  If the number of clone classes in the training set is not equal to the total number of classes. 
 - <b>`AssertionError`</b>:  If the number of clone classes is not equal to the number of weights. 
 - <b>`AssertionError`</b>:  If the number of clone classes is not equal to the number of similarity scores. 
 - <b>`AssertionError`</b>:  If the number of type classes in the training set is not equal to the total number of classes. 


---

<a href="../spaceTree/utils.py#L410"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_class_weights`

```python
compute_class_weights(y_train)
```

Calculate class weights based on the class sample count. 


---

<a href="../spaceTree/utils.py#L414"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `balanced_split`

```python
balanced_split(data, hold_in, size=0.5)
```

Splits the data into balanced train and test sets based on the given hold_in indices. 



**Parameters:**
 
 - <b>`data`</b> (object):  The data object containing the features and labels. 
 - <b>`hold_in`</b> (list):  The indices of the data to be split. 
 - <b>`size`</b> (float):  The proportion of data to be included in the test set. Default is 0.5. 



**Returns:**
 
 - <b>`train_indices_final`</b> (list):  The indices of the training set. 
 - <b>`test_indices_final`</b> (list):  The indices of the test set. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
