---
layout: default
title: "Data preprocessing"
parent: API Reference
nav_order: 2
---
<!-- markdownlint-disable -->

<a href="../spaceTree/preprocessing.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `preprocessing`



---

<a href="../spaceTree/preprocessing.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `run_scvi`

```python
run_scvi(
    adata,
    outdir='data/interim/res_scvi.csv',
    highly_variable_genes=False,
    plot_extra=[]
)
```

Runs scVI on the input AnnData object and returns a DataFrame with the cell embeddings and source labels. 



**Parameters:**
 adata (anndata.AnnData): Input AnnData object with gene expression data. outdir (str): Output directory to save the resulting DataFrame. highly_variable_genes (bool): Flag indicating whether to identify highly variable genes. plot_extra (list): List of additional variables to include in the visualization. 



**Returns:**
 pandas.DataFrame: DataFrame with cell embeddings and source labels. 


---

<a href="../spaceTree/preprocessing.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `record_edges`

```python
record_edges(emb_rna, emb_spatial, n_neigb, edge_type, metric='minkowski')
```

Create edges between nodes based on nearest neighbors. 



**Parameters:**
 emb_rna (pd.DataFrame): DataFrame containing RNA embeddings. emb_spatial (pd.DataFrame): DataFrame containing spatial embeddings. n_neigb (int): Number of nearest neighbors to consider. edge_type (str): Type of edge to create. Must be either 'sc2xen', 'sc2sc', or 'sc2vis'. metric (str, optional): Distance metric to use. Defaults to 'minkowski'. 



**Returns:**
 pd.DataFrame: DataFrame containing the edges with columns 'node1', 'node2', 'weight', and 'type'. 


---

<a href="../spaceTree/preprocessing.py#L135"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `show_weights_distribution`

```python
show_weights_distribution(
    edges,
    spatial,
    spatial_type='visium',
    library_id=None
)
```

Display the distribution of weights for each node in a spatial dataset. 



**Parameters:**
 edges (pandas.DataFrame): DataFrame containing the edges information. spatial (anndata.AnnData): Annotated data matrix containing the spatial dataset. spatial_type (str, optional): Type of spatial dataset. Defaults to "visium". library_id (str, optional): ID of the library. Defaults to None. 


---


<a href="../spaceTree/preprocessing.py#L186"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_edges_for_visium_nodes`

```python
create_edges_for_visium_nodes(visium)
```

Create edges between Visium nodes based on their spatial proximity. 



**Args:**
 
 - <b>`visium`</b> (DataFrame):  DataFrame containing Visium data. 



**Returns:**
 
 - <b>`DataFrame`</b>:  DataFrame containing the edges between Visium nodes.  The DataFrame has columns "node1", "node2", "weight", and "type". 


---

<a href="../spaceTree/preprocessing.py#L216"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_edges_for_xenium_nodes_global`

```python
create_edges_for_xenium_nodes_global(xenium, percentile=1, sample_size=1000)
```

Create edges between xenium nodes based on their centroids. 



**Parameters:**
 
- xenium (object): The xenium object containing the node data. 
- percentile (float): The percentile value used to determine the distance threshold. 
- sample_size (int): The size of the sample used to estimate the distance threshold. 



**Returns:**
 
- edges_xen_df (DataFrame): A DataFrame containing the edges between xenium nodes, along with their weights, distance threshold, and type. 


---

<a href="../spaceTree/preprocessing.py#L272"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `save_edges_and_embeddings`

```python
save_edges_and_embeddings(
    edges,
    emb_spatial,
    emb_rna,
    outdir='data/interim/',
    suffix=''
)
```

Save the edges, spatial embeddings, and RNA embeddings to CSV files. 



**Parameters:**
 edges (DataFrame): DataFrame containing the edges. emb_spatial (DataFrame): DataFrame containing the spatial embeddings. emb_rna (DataFrame): DataFrame containing the RNA embeddings. outdir (str): Directory to save the CSV files. Default is "data/interim/". suffix (str): Suffix to add to the CSV file names. Default is an empty string. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
