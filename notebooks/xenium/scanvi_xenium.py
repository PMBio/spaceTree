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
from scipy.cluster.hierarchy import dendrogram, linkage, distance, fcluster

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


#%%
sq.pl.spatial_scatter(xenium, library_id="spatial", shape=None,color=[
"total_counts"
    ],
    wspace=0.4,)
#%%
cell_types = pd.read_excel("../data/raw/Requested_Cell_Barcode_Type_Matrices.xlsx", sheet_name="scFFPE-Seq", index_col = 0)
overcl = pd.read_csv("../data/interim/clones_over.csv", index_col = 0)
overcl.columns = ["clone"]
# cell_types.index = cell_types.index.str.replace(".", "-")

cell_types = cell_types.join(overcl)
cell_types = cell_types[["Annotation","clone"]]
cell_types = cell_types.reset_index()
cell_types.columns = ["node1","cell_type","clone"]
cell_types.clone = cell_types.clone.fillna("diploid")
cell_types.clone = cell_types.clone.astype(str)
cell_types = cell_types.set_index("node1")
adata_seq.obs.index = adata_seq.obs.index.str.replace(".", "-")

adata_seq.obs = adata_seq.obs.join(cell_types[["cell_type", "clone"]])
#%%
sc.pl.umap(adata_seq, color=["cell_type", "clone"])
#%%
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
model = scvi.model.SCVI(adata, n_layers=2, n_latent=30)
model.train()
#%%
adata.obsm["X_scVI"] = model.get_latent_representation()
adata.obsm["X_mde"] = mde(adata.obsm["X_scVI"])
sc.pl.embedding(
    adata,
    basis="X_mde",
    color=["source"],
    frameon=False,
    ncols=1,
)
#%%
adata.obs.clone = adata.obs.clone.astype(str)
adata.obs.clone = adata.obs.clone.fillna("Unknown")
#%%
scanvi_model = scvi.model.SCANVI.from_scvi_model(
    model,
    adata=adata,
    labels_key="clone",
    unlabeled_category="Unknown",
)
#%%
scanvi_model.train(max_epochs=20, n_samples_per_label=100)

#%%
SCANVI_LATENT_KEY = "X_scANVI_clone"
SCANVI_PREDICTION_KEY = "C_scANVI_clone"

adata.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(adata)
adata.obs[SCANVI_PREDICTION_KEY] = scanvi_model.predict(adata)
#%%
SCANVI_MDE_KEY = "X_mde_scanvi"
adata.obsm[SCANVI_MDE_KEY] = mde(adata.obsm[SCANVI_LATENT_KEY])
#%%

adata.obs.cell_type = adata.obs.cell_type.astype(str)
adata.obs.cell_type = adata.obs.cell_type.fillna("Unknown")
#%%
scanvi_model = scvi.model.SCANVI.from_scvi_model(
    model,
    adata=adata,
    labels_key="cell_type",
    unlabeled_category="Unknown",
)
#%%
scanvi_model.train(max_epochs=20, n_samples_per_label=100)

#%%
SCANVI_LATENT_KEY = "X_scANVI_cell_type"
SCANVI_PREDICTION_KEY = "C_scANVI_cell_type"

adata.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(adata)
adata.obs[SCANVI_PREDICTION_KEY] = scanvi_model.predict(adata)
#%%
SCANVI_MDE_KEY = "X_mde_scanvi_cell_type"
adata.obsm[SCANVI_MDE_KEY] = mde(adata.obsm[SCANVI_LATENT_KEY])

#%%
#prediction = adata.obs[["C_scANVI_clone","C_scANVI_cell_type"]]
prediction = adata.obs[["C_scANVI_cell_type"]]


# %%
prediction.index = [x[:-2] for x in prediction.index]
# %%
xenium.obs = xenium.obs.join(prediction)
sq.pl.spatial_scatter(xenium, library_id="spatial", shape=None,color=[
"C_scANVI_cell_type"
    ],
    wspace=0.4,)
#%%
xen_celltype = pd.read_excel("../data/raw/Requested_Cell_Barcode_Type_Matrices.xlsx", sheet_name="Xenium R1 Fig1-5 (supervised)", index_col = 0)
#%%
xen_celltype.columns = ["cell_type_provided"]
#%%
xen_celltype.index = [str(x) for x in xen_celltype.index]
xenium.obs = xenium.obs.join(xen_celltype)
#%%
# xenium.obs = xenium.obs.join(prediction)
sq.pl.spatial_scatter(xenium, library_id="spatial", shape=None,color=
["C_scANVI_cell_type","cell_type_provided"],
    wspace=0.4,)
#%%
xenium.write_h5ad("../data/interim/scanvi_xenium.h5ad")