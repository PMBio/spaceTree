---
layout: default
title: "Dataset API"
parent: API Reference
nav_order: 3
---
<!-- markdownlint-disable -->

<a href="../spaceTree/dataset.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `dataset`





---

<a href="../spaceTree/dataset.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `validate_counts`

```python
validate_counts(counter, threshold, label)
```

Validates the counts in a counter dictionary against a threshold. 



**Args:**
 
 - <b>`counter`</b> (collections.Counter):  The counter dictionary containing the counts. 
 - <b>`threshold`</b> (int):  The minimum count threshold. 
 - <b>`label`</b> (str):  The label to be used in the assertion error message. 



**Raises:**
 
 - <b>`AssertionError`</b>:  If any count in the counter dictionary is less than the threshold. 


---

<a href="../spaceTree/dataset.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `filter_and_encode`

```python
filter_and_encode(df, node_encoder, all_nodes, use_index=False)
```

Filters and encodes the given DataFrame based on the provided node encoder and all nodes. 



**Args:**
 
 - <b>`df`</b> (pandas.DataFrame):  The DataFrame to be filtered and encoded. 
 - <b>`node_encoder`</b> (dict):  A dictionary mapping node IDs to encoded values. 
 - <b>`all_nodes`</b> (list):  A list of all node IDs. 
 - <b>`use_index`</b> (bool, optional):  Whether to filter based on DataFrame index. Defaults to False. 



**Returns:**
 
 - <b>`pandas.DataFrame`</b>:  The filtered and encoded DataFrame. 


---

<a href="../spaceTree/dataset.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `drop_small`

```python
drop_small(edges, numb)
```

Drop clones and cell types with less than 'numb' cells from the edges dataframe. 



**Parameters:**
 edges (DataFrame): The dataframe containing the edges information. numb (int): The minimum number of cells required for a clone or cell type to be included. 



**Returns:**
 DataFrame: The modified edges dataframe with small clones and cell types dropped. 


---

<a href="../spaceTree/dataset.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `preprocess_data`

```python
preprocess_data(edges, overcl, spatial_edges, grid_edges)
```

Preprocesses the given data by filtering and filling missing values. 



**Args:**
 
 - <b>`edges`</b> (DataFrame):  The edges data. 
 - <b>`overcl`</b> (DataFrame):  The annotation data with clone and cell type labels. 
 - <b>`spatial_edges`</b> (str):  The type of spatial edges. 
 - <b>`grid_edges`</b> (str):  The type of grid edges. 



**Returns:**
 
 - <b>`Tuple[DataFrame, DataFrame]`</b>:  The preprocessed edges and overcl data. 


---

<a href="../spaceTree/dataset.py#L99"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `read_and_merge_embeddings`

```python
read_and_merge_embeddings(paths, edges, drop_less=10)
```

Read and merge the embeddings from spatial and RNA datasets. 



**Parameters:**
 
- paths (dict): A dictionary containing the file paths for the spatial and RNA datasets. 
- edges (pd.DataFrame): A DataFrame containing the edges of the graph. 
- drop_less (int): The minimum number of occurrences required for an edge to be kept. 



**Returns:**
 
- emb_vis (pd.DataFrame): The merged embeddings from the spatial dataset. 
- emb_rna (pd.DataFrame): The merged embeddings from the RNA dataset. 
- edges (pd.DataFrame): The filtered edges of the graph. 
- node_encoder (dict): A dictionary mapping node IDs to encoded node IDs. 


---

<a href="../spaceTree/dataset.py#L136"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_data_object`

```python
create_data_object(
    edges,
    emb_vis,
    emb_rna,
    node_encoder,
    sim=None,
    with_diploid=True
)
```

Create a data object for graph neural network training. 



**Args:**
 
 - <b>`edges`</b> (pandas.DataFrame):  DataFrame containing the edges of the graph. 
 - <b>`emb_vis`</b> (pandas.DataFrame):  DataFrame containing the spatial embeddings. 
 - <b>`emb_rna`</b> (pandas.DataFrame):  DataFrame containing the RNA embeddings. 
 - <b>`node_encoder`</b> (dict):  Dictionary mapping node IDs to their corresponding encodings. 
 - <b>`sim`</b> (pandas.DataFrame, optional):  Similarity matrix between clone values. Defaults to None. 
 - <b>`with_diploid`</b> (bool, optional):  Flag indicating whether to include diploid values in the encoding. Defaults to True. 



**Returns:**
 
 - <b>`tuple`</b>:  A tuple containing the data object and dictionaries for node, clone, and cell type encodings.  If `sim` is provided, an additional similarity matrix is returned. 



**Raises:**
 
 - <b>`AssertionError`</b>:  If the data object is not valid or the shapes of the data arrays are not consistent. 


---

<a href="../spaceTree/dataset.py#L197"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_encoding_dict`

```python
create_encoding_dict(df, column, extras=[])
```

Create a dictionary that maps unique values in a column of a DataFrame to their corresponding indices. 



**Parameters:**
 
 - <b>`df`</b> (pandas.DataFrame):  The DataFrame containing the column. 
 - <b>`column`</b> (str):  The name of the column. 
 - <b>`extras`</b> (list, optional):  Additional values to exclude from the dictionary. 



**Returns:**
 
 - <b>`dict`</b>:  A dictionary mapping unique values to their corresponding indices. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
