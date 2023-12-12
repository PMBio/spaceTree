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
import os
os.chdir("/home/o313a/clonal_GNN/")
import re
import spaceTree.plotting as plotting
import spaceTree.utils as utils
from sklearn.metrics import f1_score,confusion_matrix

#%% Paths
path_xen= "data/raw/xenium/outs/cell_feature_matrix.h5"
path_sc = "data/interim/scrna.h5ad"
path_scvi = "" # leave emty string of needed to be recomputed
#%% Reding the data
xenium = sc.read_10x_h5(filename=path_xen)
df = pd.read_csv('data/raw/xenium/outs/cells.csv')
df.set_index(xenium.obs_names, inplace=True)
xenium.obs = df.copy()
xenium.obsm["spatial"] = xenium.obs[["x_centroid", "y_centroid"]].copy().to_numpy()
xenium.var_names_make_unique()
adata_seq = sc.read_h5ad(path_sc)
adata_seq.obs.drop(columns = ["celltype_major","celltype_minor"], inplace = True)


#%%
# plotting.plot_xenium(xenium.obs.x_centroid, xenium.obs.y_centroid, 
#             xenium.obs.cell_type, palette = ct_palette)
#%%
cell_types = pd.read_excel("data/raw/Requested_Cell_Barcode_Type_Matrices.xlsx", sheet_name="scFFPE-Seq", index_col = 0)
overcl = pd.read_csv("data/interim/clones_over.csv", index_col = 0)
overcl.columns = ["clone"]
overcl.index = [x[:-2] for x in overcl.index]
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
sc.pl.umap(adata_seq, color=["cell_type", "clone"], ncols = 1)
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
prediction = adata.obs[["C_scANVI_clone","C_scANVI_cell_type"]]
# prediction = adata.obs[["C_scANVI_cell_type"]]


# %%
prediction.index = [x[:-2] for x in prediction.index]
# %%
xenium.obs = xenium.obs.join(prediction)
#%%
plotting.plot_xenium(xenium.obs.x_centroid, xenium.obs.y_centroid, 
            xenium.obs.C_scANVI_cell_type, palette = "tab20")
#%%
annotation = pd.read_excel("data/raw/Requested_Cell_Barcode_Type_Matrices.xlsx.1", sheet_name="Xenium R1 Fig1-5 (supervised)", index_col=0)
annotation.Cluster.replace("DCIS_2", "DCIS 1", inplace=True)
annotation.Cluster.replace("DCIS_1", "DCIS 2", inplace=True)
annotation.columns = ["Annotation"]
annotation.index = [str(x) for x in annotation.index]
# %%

xenium.obs = xenium.obs.join(annotation)

#%%
list1 = list(set(xenium.obs.C_scANVI_cell_type))
list2 = list(set(xenium.obs.Annotation))
mapping_dict = {}

for name in list2:
    if isinstance(name, str):
        replaced_name = name.replace('_', ' ')
        if replaced_name in list1:
            mapping_dict[name] = replaced_name
        elif name == "Unlabeled":
            mapping_dict[name] = 'unknown'
            mapping_dict['Unlabeled'] = 'unknown'
        else:
            mapping_dict[name] = name
    else:
        mapping_dict[name] = 'unknown'
mapping_dict['DCIS 1'] = 'DCIS 2'
mapping_dict['DCIS 2'] = 'DCIS 1'

#%%
xenium.obs.Annotation = xenium.obs.Annotation.map(mapping_dict)
#%%
ct_palette = sns.color_palette("tab20", len(set(xenium.obs.Annotation)))
ct_palette = dict(zip(set(xenium.obs.Annotation), ct_palette))
#%%
plotting.plot_xenium(xenium.obs.x_centroid, xenium.obs.y_centroid, 
            xenium.obs.Annotation, palette = ct_palette)
plotting.plot_xenium(xenium.obs.x_centroid, xenium.obs.y_centroid, 
            xenium.obs.C_scANVI_cell_type, palette = ct_palette)
#%%
labels = list(set(xenium.obs.Annotation).intersection(set(xenium.obs.C_scANVI_cell_type)))
xenium_tmp = xenium.obs.copy()
xenium_tmp = xenium_tmp[xenium_tmp.Annotation.isin(labels)]
xenium_tmp = xenium_tmp[xenium_tmp.C_scANVI_cell_type.isin(labels)]
c_mat = pd.DataFrame(confusion_matrix(xenium_tmp.Annotation, xenium_tmp.C_scANVI_cell_type, labels = labels, normalize = 'true'), index = labels, columns = labels)
#%%
ax = sns.heatmap(c_mat, linewidth=.5, cmap = "Blues", annot = True, fmt=".1f")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
# %%
def f1_from_confusion(c_mat):
    # Assuming c_mat is the confusion matrix
    # For a normalized confusion matrix, diagonal elements represent recall for each class
    recall = np.diag(c_mat)
    
    # Precision for each class can be calculated as the diagonal element divided by the sum of elements in the column
    precision = np.diag(c_mat) / np.sum(c_mat, axis=0)
    
    # Calculate the F1 score for each class
    f1 = 2 * precision * recall / (precision + recall)
    
    # If you want the average F1 score, take the mean of the F1 scores for each class
    avg_f1 = np.nanmean(f1)
    
    return f1, avg_f1
f1, avg_f1 = f1_from_confusion(c_mat)
print(avg_f1)
#%%

#%%
xenium.write_h5ad("data/interim/scanvi_xenium.h5ad")
# %%
