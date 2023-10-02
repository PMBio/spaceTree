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
import squidpy as sq
def flatten(l):
    return [item for sublist in l for item in sublist]
#%% Paths
path_xen= "../data/raw/xenium/outs/cell_feature_matrix.h5"
path_sc = "../data/interim/scrna.h5ad"
path_scvi = "" # leave emty string of needed to be recomputed
#%% Reding the data
xenium = sc.read_10x_h5(filename=path_xen)
df = pd.read_csv('../data/raw/xenium/outs/cells.csv')
df.set_index(xenium.obs_names, inplace=True)
xenium.obs = df.copy()
xenium.obsm["spatial"] = xenium.obs[["x_centroid", "y_centroid"]].copy().to_numpy()
xenium.var_names_make_unique()
adata_seq = sc.read_h5ad("../data/interim/scrna.h5ad")
adata_seq.obs.drop(columns = ["celltype_major","celltype_minor"], inplace = True)
cell_types = pd.read_csv("../data/interim/cell_types.csv", index_col = 2, sep = ";")
adata_seq.obs = adata_seq.obs.join(cell_types[["celltype_major","celltype_minor"]])
adata_seq.obs.index = adata_seq.obs.index.str.replace(".", "-")
#%%
sc.pl.umap(adata_seq, color=["celltype_major", "celltype_minor"], legend_loc="on data")
#%%
sq.pl.spatial_scatter(xenium, library_id="spatial", shape=None,color=[
"total_counts"
    ],
    wspace=0.4,)
# %% Run scvi if needed
if len(path_scvi) == 0:
    adata_seq = adata_seq.raw.to_adata()
    adata_seq.X.data = np.exp(adata_seq.X.data) - 1
    adata_seq.X = adata_seq.X.multiply(adata_seq.obs.nCount_RNA.to_numpy()[:, np.newaxis]).tocsr()
    adata_seq.X = np.round(adata_seq.X / 1e4)
    adata_seq.var_names_make_unique()
    xenium.obs["source"] = "xenium"
    adata_seq.obs["source"] = "scRNA"
    adata = xenium.concatenate(adata_seq)
    sc.pp.filter_cells(adata, min_genes=3)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata  # keep full dimension safe
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
    cell_source.to_csv("../data/interim/res_scvi_xenium.csv")

# %% Create embeddings
cell_source = pd.read_csv("../data/interim/res_scvi_xenium.csv", index_col=0)
cell_source.index = [x[:-2] for x in cell_source.index]
emb_spatial = cell_source[cell_source.source == "xenium"][['0', '1']]
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
edges["type"] = "sc2xen"
edges.head()
#%% Sanity check to see which areas will be harder to map to the spatial data
xenium_tmp = xenium.copy()
top_match = edges[["node2","weight"]].groupby("node2").mean()
xenium_tmp.obs = xenium_tmp.obs.join(top_match)
sq.pl.spatial_scatter(xenium_tmp, library_id="spatial", shape=None,color=[
"weight"
    ],
    wspace=0.4,)


# %% Normalize edges weights between 0 and 1
edges.weight = (edges.weight - edges.weight.min())/(edges.weight.max() - edges.weight.min())
edges.weight  = 1 - edges.weight
# %%
dist_xen = distance.cdist(xenium.obs[["x_centroid","y_centroid"]].values, xenium.obs[["x_centroid","y_centroid"]].values, 'euclidean')
#%%
distances = [5,10,20,30]
node_list = xenium.obs.index
edges_xen = []

for i in tqdm(range(dist_xen.shape[0])):
    node_max = np.max(dist_xen[i,:])
    node_min = 0
    norm_factor = node_max - node_min
    sorted_row_indices = np.argsort(dist_xen[i])
    for d in distances:
        top_edges = sorted_row_indices[1:d+1]
        norm_values = 1 - (dist_xen[i, top_edges] - node_min) / norm_factor
        edges_xen.append([(node_list[i], node_list[j], norm_values[j],d) for j in range(d)])

#%%
edges_xen = flatten(edges_xen)
#%%
edges_xen = pd.DataFrame(edges_xen, columns = ["node1", "node2", "weight", "distance"])

edges_xen["type"] = "xen2grid"
#%%
for d in distances:
    edges_tmp = edges.copy()
    edges_tmp = pd.concat([edges_tmp,edges_xen[edges_xen.distance == d]])

    cnv_pred = pd.read_csv("../data/interim/clones_sc.csv",index_col = 0)
    cell_types = pd.read_csv("../data/interim/cell_types.csv", index_col = 2, sep = ";")
    cell_types.index = [x.replace(".", "-") for x in cell_types.index]
    cell_types = cell_types.join(cnv_pred)
    cell_types = cell_types[["celltype_major","celltype_minor","leiden"]]
    cell_types = cell_types.reset_index()
    cell_types.columns = ["node1","celltype_major","celltype_minor","clone"]
    cell_types.clone = cell_types.clone.fillna("diploid")
    cell_types.clone = cell_types.clone.astype(str)

    edges_tmp = edges_tmp.merge(cell_types, left_on = "node1", right_on = "node1", how = "left")
    edges_tmp[edges_tmp.type == "xengrid"]

    to_drop = np.where(edges_tmp[edges_tmp.type == "sc2xen"].celltype_major.isna() == True)[0]

    to_drop = edges_tmp[(edges_tmp.type == "sc2xen") & edges_tmp.celltype_major.isna()].index

    edges_tmp = edges_tmp.drop(to_drop, axis=0)


    edges_tmp.to_csv(f"../data/interim/edges_xenium{d}.csv")

# %%
emb_spatial.to_csv("../data/interim/embedding_xenium_scvi.csv")
emb_rna.to_csv("../data/interim/embedding_rna2xen_scvi.csv")



# %%
