# %%
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

#%% Paths
path_visium = "../data/raw/visium/"
path_sc = "../data/interim/scrna.h5ad"
path_scvi = "../data/interim/res_scvi.csv" # leave emty string of needed to be recomputed
#%% Reding the data
visium = sc.read_visium(path_visium, genome=None, count_file='CytAssist_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5',
                        library_id=None, load_images=True, source_image_path=None)
visium.var_names_make_unique()
coor_int = [[int(x[0]),int(x[1])] for x in visium.obsm["spatial"]]
visium.obsm["spatial"] = np.array(coor_int)
adata_seq = sc.read_h5ad("../data/interim/scrna.h5ad")
adata_seq.obs.drop(columns = ["celltype_major","celltype_minor"], inplace = True)
cell_types = pd.read_csv("../data/interim/cell_types.csv", index_col = 2, sep = ";")
adata_seq.obs = adata_seq.obs.join(cell_types[["celltype_major","celltype_minor"]])
adata_seq.obs.index = adata_seq.obs.index.str.replace(".", "-")
#%%
sc.pl.umap(adata_seq, color=["celltype_major", "celltype_minor"], legend_loc="on data")
#%%
sc.pl.spatial(visium, img_key="hires")

# %% Run scvi if needed
if len(path_scvi) == 0:
    adata_seq = adata_seq.raw.to_adata()
    adata_seq.X.data = np.exp(adata_seq.X.data) - 1
    adata_seq.X = adata_seq.X.multiply(adata_seq.obs.nCount_RNA.to_numpy()[:, np.newaxis]).tocsr()
    adata_seq.X = np.round(adata_seq.X / 1e4)
    adata_seq.var_names_make_unique()
    visium.obs["source"] = "visium"
    adata_seq.obs["source"] = "scRNA"
    adata = visium.concatenate(adata_seq)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata  # keep full dimension safe
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=2000,
        layer="counts",
        batch_key="batch",
        subset=True
    )

    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="source")
    vae = scvi.model.SCVI(adata, n_layers=2, n_latent=30)
    vae.train()
    adata.obsm["X_scVI"] = vae.get_latent_representation()
    adata.obsm["X_mde"] = mde(adata.obsm["X_scVI"])
    sc.pl.embedding(
        adata,
        basis="X_mde",
        color=["source"],
        frameon=False,
        ncols=1,
    )
    cell_source = pd.DataFrame(adata.obsm["X_mde"], index = adata.obs.index)
    cell_source["source"] = adata.obs.source
    cell_source.to_csv("../data/interim/res_scvi.csv")

# %% Create embeddings
cell_source = pd.read_csv("../data/interim/res_scvi.csv", index_col=0)
cell_source.head()
cell_source.index = [x[:-2] for x in cell_source.index]
emb_spatial = cell_source[cell_source.source == "visium"][['0', '1']]
emb_rna = cell_source[cell_source.source == "scRNA"][['0', '1']]

# %% Create distance matrix
dist = distance.cdist(emb_rna.values, emb_spatial.values, 'euclidean')
dist = pd.DataFrame(dist, index = emb_rna.index, columns = emb_spatial.index)
dist = dist.T
# %% Record the edges with top 10 nearest neighbors
n_neigb = 10
edges = []
locations = dist.columns
for spot in tqdm(dist.index):
    d = dist.loc[[spot]]
    t = np.sort(d.values)[:,n_neigb-1][0]
    indexer = (d<=t).values[0]
    loc_edges = [locations[i] for i in range(len(indexer)) if indexer[i]]
    edge_weights = [d.iloc[:,i].values[0] for i in range(len(indexer)) if indexer[i]]
    for i in range(len(loc_edges)):
        edges.append((loc_edges[i],spot,edge_weights[i]))

# %%
edges = pd.DataFrame(edges)
edges.columns = ["node1", "node2", "weight"]
edges["type"] = "sc2vis"
edges.head()
#%% Sanity check to see which areas will be harder to map to the spatial data
visium_tmp = visium.copy()
top_match = edges[["node2","weight"]].groupby("node2").mean()
visium_tmp.obs = visium_tmp.obs.join(top_match)
sc.pl.spatial(visium_tmp, cmap='magma',
                # show first 8 cell types
                color="weight",
                ncols=2, size=1.7,
                img_key='hires',
                show = True)
del visium_tmp
del top_match
# %% Normalize edges weights between 0 and 1
edges.weight = (edges.weight - edges.weight.min())/(edges.weight.max() - edges.weight.min())
edges.weight  = 1 - edges.weight
# %% 
# convert array_row and array_col columns to integers in visium.obs dataframe
visium.obs["array_row"] = visium.obs["array_row"].astype(int)
visium.obs["array_col"] = visium.obs["array_col"].astype(int)
vis_nodes = visium.obs.index
edges_vis = []
for node in tqdm(vis_nodes):
    x= visium.obs.loc[node].array_row
    y= visium.obs.loc[node].array_col
    tmp = visium.obs[(visium.obs.array_row >=x-2)&(visium.obs.array_row <=x+2)&(visium.obs.array_col <=y+2)&(visium.obs.array_col >=y-2)].copy()
    tmp.loc[:,"degree"] = 2
    tmp.loc[(tmp.array_row >=x-1)&(tmp.array_row <=x+1)&(tmp.array_col <=y+1)&(tmp.array_col >=y-1),"degree"] = 1
    nodes1 = tmp[tmp.degree == 1].index
    for n in nodes1:
        if n != node:
            edges_vis.append([node,n, 1])
    nodes2 = tmp[tmp.degree == 2].index
    for n in nodes2:
        edges_vis.append([node,n, 0.5])

# %%
edges_vis = pd.DataFrame(edges_vis, columns = ["node1", "node2", "weight"])

# %%
edges_vis["type"] = "vis2grid"

# %%
edges = pd.concat([edges,edges_vis])

# %% [markdown]
# # 3. Clones and cell types from RNA
cnv_pred = pd.read_csv("../data/interim/clones_sc.csv",index_col = 0)
cell_types.index = [x.replace(".", "-") for x in cell_types.index]
cell_types = cell_types.join(cnv_pred)
cell_types = cell_types[["celltype_major","celltype_minor","leiden"]]
cell_types = cell_types.reset_index()
cell_types.columns = ["node1","celltype_major","celltype_minor","clone"]
cell_types.clone = cell_types.clone.fillna("diploid")
cell_types.clone = cell_types.clone.astype(str)



# %%
edges_tmp = edges.copy()
edges_tmp
edges_tmp = edges_tmp.merge(cell_types, left_on = "node1", right_on = "node1", how = "left")
edges_tmp[edges_tmp.type == "vis2grid"]
edges_tmp

# %%
to_drop = np.where(edges_tmp[edges_tmp.type == "sc2vis"].celltype_major.isna() == True)[0]

to_drop = edges_tmp[(edges_tmp.type == "sc2vis") & edges_tmp.celltype_major.isna()].index

# %%
edges_tmp = edges_tmp.drop(to_drop, axis=0)


# %%
edges_tmp.to_csv("../data/interim/edges.csv")

# %%
emb_spatial.to_csv("../data/interim/embedding_visium_scvi.csv")
emb_rna.to_csv("../data/interim/embedding_rna2vis_scvi.csv")



# %%
