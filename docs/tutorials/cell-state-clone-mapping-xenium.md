---
layout: default
title: "Cell State and Clone Mapping to 10x Xenium with spaceTree"
parent: Tutorials
nav_order: 2
---

# Cell State and Clone Mapping to 10x Xenium with spaceTree

This tutorial is based on public data from Janesick et al. 2023: [High resolution mapping of the tumor microenvironment using integrated single-cell, spatial and in situ analysis](https://www.nature.com/articles/s41467-023-43458-x).

In particular, we will use the following files:

- Xenium output bundle (please note that 10x only allows to download the whole bundle which takes 9.86 GB)
- FRP (scRNA) HDF5 file
- Annotation files:

    - Cell Type annotation file
    - Clone annotation file (based on [infercnvpy](https://github.com/icbi-lab/infercnvpy) run on FRP data) (provided in the `data` folder). We also provide a [tutorial](https://github.com/PMBio/spaceTree/blob/master/notebooks/infercnv_run.ipynb) on how to generate this file.

All data should be downloaded and placed in the `data` folder. You should download the data using the following commands:

```bash
cd data/
# annotation file
wget https://cdn.10xgenomics.com/raw/upload/v1695234604/Xenium%20Preview%20Data/Cell_Barcode_Type_Matrices.xlsx
# scFFPE data
wget https://cf.10xgenomics.com/samples/spatial-exp/2.0.0/CytAssist_FFPE_Human_Breast_Cancer/CytAssist_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5
# Xenium bundle (9.86 GB)
wget https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip
unzip Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip
```

## 0: Imports


```python
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm
from scipy.spatial import distance
import scvi
from scvi.model.utils import mde
import os
import spaceTree.preprocessing as pp
import spaceTree.dataset as dataset
import warnings
import re
import pickle
import torch
from torch_geometric.loader import DataLoader,NeighborLoader
import torch.nn.functional as F
import lightning.pytorch as pl
import spaceTree.utils as utils
import spaceTree.plotting as sp_plot
from spaceTree.models import *
warnings.simplefilter("ignore")
```

## 1: Prepare the data for spaceTree

### 1.1: Open data files


```python
adata_ref = sc.read_10x_h5('../data/Chromium_FFPE_Human_Breast_Cancer_Chromium_FFPE_Human_Breast_Cancer_count_sample_filtered_feature_bc_matrix.h5')
adata_ref.var_names_make_unique()
cell_type = pd.read_excel("../data/Cell_Barcode_Type_Matrices.xlsx", sheet_name="scFFPE-Seq", index_col = 0)
clone_anno = pd.read_csv("../data/clone_annotation.csv", index_col = 0)
```


```python
adata_ref.obs = adata_ref.obs.join(cell_type, how="left").join(clone_anno, how="left")
```


```python
adata_ref.obs.columns = ['cell_type', 'clone']
adata_ref = adata_ref[~adata_ref.obs.cell_type.isna()]
```


```python
xenium = sc.read_10x_h5(filename='../data/outs/cell_feature_matrix.h5')
cell_df = pd.read_csv('../data/outs/cells.csv.gz')
cell_df.set_index(xenium.obs_names, inplace=True)
xenium.obs = cell_df.copy()
xenium.obsm["spatial"] = xenium.obs[["x_centroid", "y_centroid"]].copy().to_numpy()
xenium.var_names_make_unique()
```


```python
xenium.obs

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cell_id</th>
      <th>x_centroid</th>
      <th>y_centroid</th>
      <th>transcript_counts</th>
      <th>control_probe_counts</th>
      <th>control_codeword_counts</th>
      <th>total_counts</th>
      <th>cell_area</th>
      <th>nucleus_area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>847.259912</td>
      <td>326.191365</td>
      <td>28</td>
      <td>1</td>
      <td>0</td>
      <td>29</td>
      <td>58.387031</td>
      <td>26.642187</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>826.341995</td>
      <td>328.031830</td>
      <td>94</td>
      <td>0</td>
      <td>0</td>
      <td>94</td>
      <td>197.016719</td>
      <td>42.130781</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>848.766919</td>
      <td>331.743187</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>16.256250</td>
      <td>12.688906</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>824.228409</td>
      <td>334.252643</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>42.311406</td>
      <td>10.069844</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>841.357538</td>
      <td>332.242505</td>
      <td>48</td>
      <td>0</td>
      <td>0</td>
      <td>48</td>
      <td>107.652500</td>
      <td>37.479687</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>167776</th>
      <td>167776</td>
      <td>7455.475342</td>
      <td>5114.875415</td>
      <td>229</td>
      <td>1</td>
      <td>0</td>
      <td>230</td>
      <td>220.452812</td>
      <td>60.599688</td>
    </tr>
    <tr>
      <th>167777</th>
      <td>167777</td>
      <td>7483.727051</td>
      <td>5111.477490</td>
      <td>79</td>
      <td>0</td>
      <td>0</td>
      <td>79</td>
      <td>37.389375</td>
      <td>25.242344</td>
    </tr>
    <tr>
      <th>167778</th>
      <td>167778</td>
      <td>7470.159424</td>
      <td>5119.132056</td>
      <td>397</td>
      <td>0</td>
      <td>0</td>
      <td>397</td>
      <td>287.058281</td>
      <td>86.700000</td>
    </tr>
    <tr>
      <th>167779</th>
      <td>167779</td>
      <td>7477.737207</td>
      <td>5128.712817</td>
      <td>117</td>
      <td>0</td>
      <td>0</td>
      <td>117</td>
      <td>235.354375</td>
      <td>25.197187</td>
    </tr>
    <tr>
      <th>167780</th>
      <td>167780</td>
      <td>7489.376562</td>
      <td>5123.197778</td>
      <td>378</td>
      <td>0</td>
      <td>0</td>
      <td>378</td>
      <td>270.079531</td>
      <td>111.806875</td>
    </tr>
  </tbody>
</table>
<p>167780 rows × 9 columns</p>
</div>



### 1.1: Filter Xenium data


```python
sc.pp.calculate_qc_metrics(xenium, percent_top=(10, 20, 50, 150), inplace=True)
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

axs[0].set_title("Total transcripts per cell")
sns.histplot(
    xenium.obs["total_counts"],
    kde=False,
    ax=axs[0],
)

axs[1].set_title("Unique transcripts per cell")
sns.histplot(
    xenium.obs["n_genes_by_counts"],
    kde=False,
    ax=axs[1],
)

```




    <Axes: title={'center': 'Unique transcripts per cell'}, xlabel='n_genes_by_counts', ylabel='Count'>




    
![png](cell-state-clone-mapping-xenium_files/cell-state-clone-mapping-xenium_12_1.png)
    



```python
# Filter the data
sc.pp.filter_cells(xenium, min_counts=10)
sc.pp.filter_genes(xenium, min_cells=5)
```


```python
xenium.obs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cell_id</th>
      <th>x_centroid</th>
      <th>y_centroid</th>
      <th>transcript_counts</th>
      <th>control_probe_counts</th>
      <th>control_codeword_counts</th>
      <th>total_counts</th>
      <th>cell_area</th>
      <th>nucleus_area</th>
      <th>n_genes_by_counts</th>
      <th>log1p_n_genes_by_counts</th>
      <th>log1p_total_counts</th>
      <th>pct_counts_in_top_10_genes</th>
      <th>pct_counts_in_top_20_genes</th>
      <th>pct_counts_in_top_50_genes</th>
      <th>pct_counts_in_top_150_genes</th>
      <th>n_counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>847.259912</td>
      <td>326.191365</td>
      <td>28</td>
      <td>1</td>
      <td>0</td>
      <td>28.0</td>
      <td>58.387031</td>
      <td>26.642187</td>
      <td>15</td>
      <td>2.772589</td>
      <td>3.367296</td>
      <td>82.142857</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>826.341995</td>
      <td>328.031830</td>
      <td>94</td>
      <td>0</td>
      <td>0</td>
      <td>94.0</td>
      <td>197.016719</td>
      <td>42.130781</td>
      <td>38</td>
      <td>3.663562</td>
      <td>4.553877</td>
      <td>54.255319</td>
      <td>79.787234</td>
      <td>100.000000</td>
      <td>100.0</td>
      <td>94.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>824.228409</td>
      <td>334.252643</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>11.0</td>
      <td>42.311406</td>
      <td>10.069844</td>
      <td>9</td>
      <td>2.302585</td>
      <td>2.484907</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>841.357538</td>
      <td>332.242505</td>
      <td>48</td>
      <td>0</td>
      <td>0</td>
      <td>48.0</td>
      <td>107.652500</td>
      <td>37.479687</td>
      <td>33</td>
      <td>3.526361</td>
      <td>3.891820</td>
      <td>52.083333</td>
      <td>72.916667</td>
      <td>100.000000</td>
      <td>100.0</td>
      <td>48.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>835.284583</td>
      <td>338.135696</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>10.0</td>
      <td>56.851719</td>
      <td>17.701250</td>
      <td>8</td>
      <td>2.197225</td>
      <td>2.397895</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>167776</th>
      <td>167776</td>
      <td>7455.475342</td>
      <td>5114.875415</td>
      <td>229</td>
      <td>1</td>
      <td>0</td>
      <td>229.0</td>
      <td>220.452812</td>
      <td>60.599688</td>
      <td>77</td>
      <td>4.356709</td>
      <td>5.438079</td>
      <td>42.794760</td>
      <td>63.318777</td>
      <td>88.209607</td>
      <td>100.0</td>
      <td>229.0</td>
    </tr>
    <tr>
      <th>167777</th>
      <td>167777</td>
      <td>7483.727051</td>
      <td>5111.477490</td>
      <td>79</td>
      <td>0</td>
      <td>0</td>
      <td>79.0</td>
      <td>37.389375</td>
      <td>25.242344</td>
      <td>37</td>
      <td>3.637586</td>
      <td>4.382027</td>
      <td>55.696203</td>
      <td>78.481013</td>
      <td>100.000000</td>
      <td>100.0</td>
      <td>79.0</td>
    </tr>
    <tr>
      <th>167778</th>
      <td>167778</td>
      <td>7470.159424</td>
      <td>5119.132056</td>
      <td>397</td>
      <td>0</td>
      <td>0</td>
      <td>397.0</td>
      <td>287.058281</td>
      <td>86.700000</td>
      <td>75</td>
      <td>4.330733</td>
      <td>5.986452</td>
      <td>55.415617</td>
      <td>73.551637</td>
      <td>93.702771</td>
      <td>100.0</td>
      <td>397.0</td>
    </tr>
    <tr>
      <th>167779</th>
      <td>167779</td>
      <td>7477.737207</td>
      <td>5128.712817</td>
      <td>117</td>
      <td>0</td>
      <td>0</td>
      <td>117.0</td>
      <td>235.354375</td>
      <td>25.197187</td>
      <td>51</td>
      <td>3.951244</td>
      <td>4.770685</td>
      <td>47.863248</td>
      <td>69.230769</td>
      <td>99.145299</td>
      <td>100.0</td>
      <td>117.0</td>
    </tr>
    <tr>
      <th>167780</th>
      <td>167780</td>
      <td>7489.376562</td>
      <td>5123.197778</td>
      <td>378</td>
      <td>0</td>
      <td>0</td>
      <td>378.0</td>
      <td>270.079531</td>
      <td>111.806875</td>
      <td>77</td>
      <td>4.356709</td>
      <td>5.937536</td>
      <td>44.444444</td>
      <td>69.312169</td>
      <td>92.857143</td>
      <td>100.0</td>
      <td>378.0</td>
    </tr>
  </tbody>
</table>
<p>164000 rows × 17 columns</p>
</div>



Lightwweight data visualization function:


```python
#Visualize the data
sp_plot.plot_xenium(xenium.obs.x_centroid, xenium.obs.y_centroid, 
            xenium.obs.total_counts, palette="viridis")
```


    
![png](cell-state-clone-mapping-xenium_files/cell-state-clone-mapping-xenium_16_0.png)
    


### 1.2: Run scvi to remove batch effects and prepare data for knn-graph construction


```python
xenium.obs["source"] = "spatial"
adata_ref.obs["source"] = "scRNA"
adata = xenium.concatenate(adata_ref)
cell_source = pp.run_scvi(adata, "../data/res_scvi_xenium.csv")
```

    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    You are using a CUDA device ('NVIDIA GeForce RTX 4090') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [1]


    Epoch 42/42: 100%|██████████| 42/42 [05:04<00:00,  7.24s/it, v_num=1, train_loss_step=181, train_loss_epoch=161]

    `Trainer.fit` stopped: `max_epochs=42` reached.


    Epoch 42/42: 100%|██████████| 42/42 [05:04<00:00,  7.24s/it, v_num=1, train_loss_step=181, train_loss_epoch=161]



    
![png](cell-state-clone-mapping-xenium_files/cell-state-clone-mapping-xenium_18_4.png)
    



```python
cell_source.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1-0</th>
      <td>1.078594</td>
      <td>-0.354043</td>
      <td>spatial</td>
    </tr>
    <tr>
      <th>2-0</th>
      <td>0.826982</td>
      <td>-0.345308</td>
      <td>spatial</td>
    </tr>
    <tr>
      <th>4-0</th>
      <td>0.926901</td>
      <td>0.060634</td>
      <td>spatial</td>
    </tr>
    <tr>
      <th>5-0</th>
      <td>0.743756</td>
      <td>-0.127475</td>
      <td>spatial</td>
    </tr>
    <tr>
      <th>7-0</th>
      <td>0.701196</td>
      <td>-0.237537</td>
      <td>spatial</td>
    </tr>
  </tbody>
</table>
</div>




```python
cell_source.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TTTGTGAGTGGTTACT-1</th>
      <td>0.708244</td>
      <td>1.117670</td>
      <td>scRNA</td>
    </tr>
    <tr>
      <th>TTTGTGAGTGTTCCAG-1</th>
      <td>-0.452168</td>
      <td>0.293400</td>
      <td>scRNA</td>
    </tr>
    <tr>
      <th>TTTGTGAGTTACTTCT-1</th>
      <td>0.786309</td>
      <td>1.310445</td>
      <td>scRNA</td>
    </tr>
    <tr>
      <th>TTTGTGAGTTGTCATA-1</th>
      <td>0.680145</td>
      <td>-0.941370</td>
      <td>scRNA</td>
    </tr>
    <tr>
      <th>TTTGTGAGTTTGGCCA-1</th>
      <td>0.865912</td>
      <td>0.308610</td>
      <td>scRNA</td>
    </tr>
  </tbody>
</table>
</div>



### 1.3: Construct the knn-graphs


```python
# get rid of the number in the end of the index
cell_source.index = [re.sub(r'-(\d+).*', r'-\1', s) for s in cell_source.index]
emb_spatial =cell_source[cell_source.source == "spatial"][[0,1]]
emb_spatial.index = [x.split("-")[0] for x in emb_spatial.index]
emb_rna = cell_source[cell_source.source == "scRNA"][[0,1]]
```


```python
print("1. Recording edges between RNA and spatial embeddings...")
# 10 here is the number of neighbors to be considered
edges_sc2xen = pp.record_edges(emb_spatial, emb_rna,10, "sc2xen")
# checking where we have the highest distance to refrence data as those might be potentially problematic spots for mapping
edges_sc2xen = pp.normalize_edge_weights(edges_sc2xen)

print("2. Recording edges between RNA embeddings...")
# 10 here is the number of neighbors to be considered
edges_sc2sc = pp.record_edges(emb_rna,emb_rna, 10, "sc2sc")
edges_sc2sc = pp.normalize_edge_weights(edges_sc2sc)
```

    1. Recording edges between RNA and spatial embeddings...
    2. Recording edges between RNA embeddings...



```python
print("3. Creating edges for Visium nodes...")
edges_xen = pp.create_edges_for_xenium_nodes_global(xenium,10)

print("4. Saving edges and embeddings...")
edges = pd.concat([edges_sc2xen, edges_sc2sc, edges_xen])
edges.node1 = edges.node1.astype(str)
edges.node2 = edges.node2.astype(str)

pp.save_edges_and_embeddings(edges, emb_spatial, emb_rna, outdir ="../data/tmp/", suffix="xenium")
```

    3. Creating edges for Visium nodes...
    4. Saving edges and embeddings...



```python
edges.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>node1</th>
      <th>node2</th>
      <th>weight</th>
      <th>type</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GATTACCGTGACCTAT-1</td>
      <td>1</td>
      <td>0.96229</td>
      <td>sc2xen</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TTTGCGCAGACTCAAA-1</td>
      <td>1</td>
      <td>0.961565</td>
      <td>sc2xen</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CCATCCCTCAATCCTG-1</td>
      <td>1</td>
      <td>0.961372</td>
      <td>sc2xen</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TATACCTAGTTTACCG-1</td>
      <td>1</td>
      <td>0.954399</td>
      <td>sc2xen</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CGTAAACAGTGATTAG-1</td>
      <td>1</td>
      <td>0.948677</td>
      <td>sc2xen</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Make sure that we have 3 types of edges:
- single-cell (reference) to visium `sc2xen`
- visium to visium `xen2grid`
- single-cell (reference) to single-cell (reference) `sc2sc`


```python
edges["type"].unique()
```




    array(['sc2xen', 'sc2sc', 'xen2grid'], dtype=object)



### 1.4: Create the dataset object for pytorch

For the next step we need to convert node IDs and classes (cell types and clones) into numerial values that can be further used by the model 


```python
#annotation file
annotations = adata_ref.obs[["cell_type", "clone"]].copy().reset_index()
annotations.columns = ["node1", "cell_type", "clone"]
annotations.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>node1</th>
      <th>cell_type</th>
      <th>clone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AAACAAGCAAACGGGA-1</td>
      <td>Stromal</td>
      <td>diploid</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AAACAAGCAAATAGGA-1</td>
      <td>Macrophages 1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AAACAAGCAACAAGTT-1</td>
      <td>Perivascular-Like</td>
      <td>diploid</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AAACAAGCAACCATTC-1</td>
      <td>Myoepi ACTA2+</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AAACAAGCAACTAAAC-1</td>
      <td>Myoepi ACTA2+</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# first we ensure that there are no missing values and combine annotations with the edges dataframe
edges_enc, annotations_enc = dataset.preprocess_data(edges, annotations,"sc2xen","xen2grid")

```


```python
edges_enc.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>node1</th>
      <th>node2</th>
      <th>weight</th>
      <th>type</th>
      <th>distance</th>
      <th>clone</th>
      <th>cell_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GATTACCGTGACCTAT-1</td>
      <td>1</td>
      <td>0.962290</td>
      <td>sc2xen</td>
      <td>NaN</td>
      <td>4</td>
      <td>DCIS 2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TTTGCGCAGACTCAAA-1</td>
      <td>1</td>
      <td>0.961565</td>
      <td>sc2xen</td>
      <td>NaN</td>
      <td>4</td>
      <td>DCIS 2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CCATCCCTCAATCCTG-1</td>
      <td>1</td>
      <td>0.961372</td>
      <td>sc2xen</td>
      <td>NaN</td>
      <td>2</td>
      <td>DCIS 1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TATACCTAGTTTACCG-1</td>
      <td>1</td>
      <td>0.954399</td>
      <td>sc2xen</td>
      <td>NaN</td>
      <td>4</td>
      <td>DCIS 2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CGTAAACAGTGATTAG-1</td>
      <td>1</td>
      <td>0.948677</td>
      <td>sc2xen</td>
      <td>NaN</td>
      <td>4</td>
      <td>DCIS 2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# specify paths to the embeddings that we will use as features for the nodes. Please don't modify unless you previously saved the embeddings in a different location
embedding_paths = {"spatial":f"../data/tmp/embedding_spatial_xenium.csv",
                    "rna":f"../data/tmp/embedding_rna_xenium.csv"}
```


```python
#next we encode all strings as ingeres and ensure consistancy between the edges and the annotations
emb_xen_nodes, emb_rna_nodes, edges_enc, node_encoder = dataset.read_and_merge_embeddings(embedding_paths, edges_enc)

```

    Excluding 0 clones with less than 10 cells
    Excluding 0 cell types with less than 10 cells



```python
edges_enc.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>node1</th>
      <th>node2</th>
      <th>weight</th>
      <th>type</th>
      <th>distance</th>
      <th>clone</th>
      <th>cell_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>82472</td>
      <td>10242</td>
      <td>0.962290</td>
      <td>sc2xen</td>
      <td>NaN</td>
      <td>4</td>
      <td>DCIS 2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>93821</td>
      <td>10242</td>
      <td>0.961565</td>
      <td>sc2xen</td>
      <td>NaN</td>
      <td>4</td>
      <td>DCIS 2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>149960</td>
      <td>10242</td>
      <td>0.961372</td>
      <td>sc2xen</td>
      <td>NaN</td>
      <td>2</td>
      <td>DCIS 1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>81201</td>
      <td>10242</td>
      <td>0.954399</td>
      <td>sc2xen</td>
      <td>NaN</td>
      <td>4</td>
      <td>DCIS 2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7770</td>
      <td>10242</td>
      <td>0.948677</td>
      <td>sc2xen</td>
      <td>NaN</td>
      <td>4</td>
      <td>DCIS 2</td>
    </tr>
  </tbody>
</table>
</div>




```python
#make sure that weight is a float
edges_enc.weight = edges_enc.weight.astype(float)

```


```python
#Finally creating a pytorch dataset object and a dictionaru that will be used for decoding the data
data, encoding_dict = dataset.create_data_object(edges_enc, emb_vis_nodes, emb_rna_nodes, node_encoder)

```


```python
torch.save(data, "../data/tmp/data_xenium.pt")
with open('../data/tmp/full_encoding_xenium.pkl', 'wb') as fp:
    pickle.dump(encoding_dict, fp)
```

## 2: Running spaceTree

### 2.1: Load the data and ID encoder/decoder dictionaries


```python
data = torch.load("../data/tmp/data_xenium.pt")
with open('../data/tmp/full_encoding_xenium.pkl', 'rb') as handle:
    encoder_dict = pickle.load(handle)
node_encoder_rev = {val:key for key,val in encoder_dict["nodes"].items()}
node_encoder_clone = {val:key for key,val in encoder_dict["clones"].items()}
node_encoder_ct = {val:key for key,val in encoder_dict["types"].items()}
data.edge_attr = data.edge_attr.reshape((-1,1))

```

### 2.2: Separate spatial nodes from reference nodes


```python

hold_out_indices = np.where(data.y_clone == -1)[0]
hold_out = torch.tensor(hold_out_indices, dtype=torch.long)

total_size = data.x.shape[0] - len(hold_out)
train_size = int(0.8 * total_size)

# Get indices that are not in hold_out
hold_in_indices = np.arange(data.x.shape[0])
hold_in = [index for index in hold_in_indices if index not in hold_out]
```

### 2.3: Create test set from reference nodes


```python
# Split the data into train and test sets
train_indices, test_indices = utils.balanced_split(data,hold_in, size = 0.3)

# Assign the indices to data masks
data.train_mask = torch.tensor(train_indices, dtype=torch.long)
data.test_mask = torch.tensor(test_indices, dtype=torch.long)

# Set the hold_out data
data.hold_out = hold_out
```

### 2.3: Create weights for the NLL loss to ensure that the model learns the correct distribution of cell types and clones


```python
y_train_type = data.y_type[data.train_mask]
weight_type_values = utils.compute_class_weights(y_train_type)
weight_type = torch.tensor(weight_type_values, dtype=torch.float)
y_train_clone = data.y_clone[data.train_mask]
weight_clone_values = utils.compute_class_weights(y_train_clone)
weight_clone = torch.tensor(weight_clone_values, dtype=torch.float)
data.num_classes_clone = len(data.y_clone.unique())
data.num_classes_type = len(data.y_type.unique())
```

### 2.4: Create Neigborhor Loader for efficient training


```python
del data.edge_type

train_loader = NeighborLoader(
    data,
    num_neighbors=[10] * 3,
    batch_size=128,input_nodes = data.train_mask
)

valid_loader = NeighborLoader(
    data,
    num_neighbors=[10] * 3,
    batch_size=128,input_nodes = data.test_mask
)
```

### 2.5: Specifying the device and sending the data to the device


```python
device = torch.device('cuda:0')
data = data.to(device)
weight_clone =  weight_clone.to(device)
weight_type = weight_type.to(device)
data.num_classes_clone = len(data.y_clone.unique())
data.num_classes_type = len(data.y_type.unique())

```

### 2.6: Model specification and training


```python
lr = 0.01
hid_dim = 100
head = 2
wd = 0.001
model = GATLightningModule_sampler(data, 
                                   weight_clone, weight_type, learning_rate=lr, 
                                   heads = head, dim_h = hid_dim, weight_decay= wd)
model = model.to(device)
early_stop_callback = pl.callbacks.EarlyStopping(monitor="validation_combined_loss", min_delta=1e-4, patience=10, verbose=True, mode="min")
trainer1 = pl.Trainer(max_epochs=1000, accelerator = "gpu", devices = [0],
                    callbacks = [early_stop_callback], 
                    log_every_n_steps=10)

```

    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs



```python
trainer1.fit(model, train_loader, valid_loader)
```

    You are using a CUDA device ('NVIDIA GeForce RTX 4090') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [1]
    
      | Name  | Type | Params
    -------------------------------
    0 | model | GAT2 | 55.9 K
    -------------------------------
    55.9 K    Trainable params
    0         Non-trainable params
    55.9 K    Total params
    0.224     Total estimated model params size (MB)


    Epoch 0: 100%|██████████| 151/151 [00:01<00:00, 79.98it/s, v_num=1, validation_acc_clone=0.663, validation_acc_ct=0.616, validation_combined_loss=0.000856, train_combined_loss=1.540]

    Metric validation_combined_loss improved. New best score: 0.001


    Epoch 2: 100%|██████████| 151/151 [00:01<00:00, 79.97it/s, v_num=1, validation_acc_clone=0.679, validation_acc_ct=0.673, validation_combined_loss=0.000735, train_combined_loss=1.160] 

    Metric validation_combined_loss improved by 0.000 >= min_delta = 0.0001. New best score: 0.001


    Epoch 12: 100%|██████████| 151/151 [00:01<00:00, 89.75it/s, v_num=1, validation_acc_clone=0.711, validation_acc_ct=0.725, validation_combined_loss=0.000632, train_combined_loss=1.010] 

    Metric validation_combined_loss improved by 0.000 >= min_delta = 0.0001. New best score: 0.001


    Epoch 22: 100%|██████████| 151/151 [00:01<00:00, 86.02it/s, v_num=1, validation_acc_clone=0.715, validation_acc_ct=0.730, validation_combined_loss=0.000607, train_combined_loss=0.994] 

    Monitored metric validation_combined_loss did not improve in the last 10 records. Best score: 0.001. Signaling Trainer to stop.


    Epoch 22: 100%|██████████| 151/151 [00:01<00:00, 85.77it/s, v_num=1, validation_acc_clone=0.715, validation_acc_ct=0.730, validation_combined_loss=0.000607, train_combined_loss=0.994]
    



```python
# Predction on spatial data
model.eval()
model = model.to(device)
with torch.no_grad():
    out, w, _ = model(data)
```


```python
# Decoding the results back to the original format
clone_res,ct_res= utils.get_calibrated_results(out, data, node_encoder_rev, node_encoder_ct,node_encoder_clone, 1)
```

## 3: Results and visualization


```python
clone_res["clone"] = clone_res.idxmax(axis=1)
ct_res["cell_type"] = ct_res.idxmax(axis=1)
```


```python
xenium.obs = xenium.obs.join(clone_res[["clone"]]).join(ct_res[["cell_type"]])
xenium.obs.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cell_id</th>
      <th>x_centroid</th>
      <th>y_centroid</th>
      <th>transcript_counts</th>
      <th>control_probe_counts</th>
      <th>control_codeword_counts</th>
      <th>total_counts</th>
      <th>cell_area</th>
      <th>nucleus_area</th>
      <th>n_genes_by_counts</th>
      <th>log1p_n_genes_by_counts</th>
      <th>log1p_total_counts</th>
      <th>pct_counts_in_top_10_genes</th>
      <th>pct_counts_in_top_20_genes</th>
      <th>pct_counts_in_top_50_genes</th>
      <th>pct_counts_in_top_150_genes</th>
      <th>n_counts</th>
      <th>source</th>
      <th>clone</th>
      <th>cell_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>847.259912</td>
      <td>326.191365</td>
      <td>28</td>
      <td>1</td>
      <td>0</td>
      <td>28.0</td>
      <td>58.387031</td>
      <td>26.642187</td>
      <td>15</td>
      <td>2.772589</td>
      <td>3.367296</td>
      <td>82.142857</td>
      <td>100.000000</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>28.0</td>
      <td>spatial</td>
      <td>4</td>
      <td>DCIS 2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>826.341995</td>
      <td>328.031830</td>
      <td>94</td>
      <td>0</td>
      <td>0</td>
      <td>94.0</td>
      <td>197.016719</td>
      <td>42.130781</td>
      <td>38</td>
      <td>3.663562</td>
      <td>4.553877</td>
      <td>54.255319</td>
      <td>79.787234</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>94.0</td>
      <td>spatial</td>
      <td>4</td>
      <td>DCIS 2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>824.228409</td>
      <td>334.252643</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>11.0</td>
      <td>42.311406</td>
      <td>10.069844</td>
      <td>9</td>
      <td>2.302585</td>
      <td>2.484907</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>11.0</td>
      <td>spatial</td>
      <td>4</td>
      <td>DCIS 2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>841.357538</td>
      <td>332.242505</td>
      <td>48</td>
      <td>0</td>
      <td>0</td>
      <td>48.0</td>
      <td>107.652500</td>
      <td>37.479687</td>
      <td>33</td>
      <td>3.526361</td>
      <td>3.891820</td>
      <td>52.083333</td>
      <td>72.916667</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>48.0</td>
      <td>spatial</td>
      <td>4</td>
      <td>DCIS 2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>835.284583</td>
      <td>338.135696</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>10.0</td>
      <td>56.851719</td>
      <td>17.701250</td>
      <td>8</td>
      <td>2.197225</td>
      <td>2.397895</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>10.0</td>
      <td>spatial</td>
      <td>4</td>
      <td>DCIS 2</td>
    </tr>
  </tbody>
</table>
</div>



### 3.1: Clon mapping
First we will visualize the clone mapping results and compare them to the histological annotation provided by the authors:
<div style="text-align: left;">
  <a href="https://raw.githubusercontent.com/PMBio/spaceTree/master/docs/histo.png" download>
    <img src="https://raw.githubusercontent.com/PMBio/spaceTree/master/docs/histo.png" alt="histo" width="200"/>
  </a>
</div>
(note that the images are not fully overlapping)


```python
sp_plot.plot_xenium(xenium.obs.x_centroid, xenium.obs.y_centroid, 
            xenium.obs.clone)
```


    
![png](cell-state-clone-mapping-xenium_files/cell-state-clone-mapping-xenium_61_0.png)
    


We can see that invasive tumor alligns with clone 0, DCIS 1 with clone 2, DCIS 2 with clone 3 (top part) and 4 (right part).
Interestingly, that DCIS 2 was separated into two clones with distinct spatial locations. Indeed, despite being classified as a single DCIS, those clones have distinct CNV patterns (e.g. chr3 and chr19):
<div style="text-align: left;">
  <a href="https://raw.githubusercontent.com/PMBio/spaceTree/master/docs/cnv_map.png" download>
    <img src="https://raw.githubusercontent.com/PMBio/spaceTree/master/docs/cnv_map.png" alt="histo" width="2000"/>
  </a>
</div>

(the image is taken from `../docs//infercnv_run.ipynb`)

Moreover: the clonal mapping perfectly alligns to the one obtained from Visium [spaceTree tutorial](https://pmbio.github.io/spaceTree/) despite vast differences in the data types and resolution.
<div style="text-align: left;">
  <a href="https://github.com/PMBio/spaceTree/blob/master/docs/tutorials/cell-state-clone-mapping_files/cell-state-clone-mapping_55_0.png?raw=true" download>
    <img src="https://github.com/PMBio/spaceTree/blob/master/docs/tutorials/cell-state-clone-mapping_files/cell-state-clone-mapping_55_0.png?raw=true" alt="histo" width="2000"/>
  </a>
</div>


Cell types can be visualized in the same way:


```python
sp_plot.plot_xenium(xenium.obs.x_centroid, xenium.obs.y_centroid, 
            xenium.obs.cell_type)
```


    
![png](cell-state-clone-mapping-xenium_files/cell-state-clone-mapping-xenium_65_0.png)
    



```python

```
