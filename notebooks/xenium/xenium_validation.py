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

import matplotlib as mpl
from scipy.cluster.hierarchy import dendrogram, linkage, distance, fcluster


def rotate_90_degrees_clockwise(matrix):
    min_x, min_y = matrix.min(axis=0)
    max_x, max_y = matrix.max(axis=0)
    
    w = max_x - min_x
    h = max_y - min_y
    # Translate to center
    matrix[:, 0] -= w/2
    matrix[:, 1] -= h/2

    # Rotate
    rotated = np.zeros_like(matrix)
    rotated[:, 0] = -matrix[:, 1]
    rotated[:, 1] = matrix[:, 0]

    # Translate back
    rotated[:, 0] += h/2
    rotated[:, 1] += w/2
    
    return rotated

#%% Paths
path_xen= "../data/interim/scanvi_xenium.h5ad"
xen_cr = pd.read_excel("../data/raw/Requested_Cell_Barcode_Type_Matrices.xlsx.1", sheet_name="Xenium R1 Fig3 (unsupervised)")

#%% Reding the data
xenium = sc.read_h5ad(filename=path_xen)

xenium.obsm["spatial"] = rotate_90_degrees_clockwise(xenium.obs[["x_centroid", "y_centroid"]].copy().to_numpy())
xenium.var_names_make_unique()

# %%
clone_res = pd.read_csv("../data/interim/clone_pred_xen.csv", index_col=0)
ct_res = pd.read_csv("../data/interim/ct_pred_xen.csv",index_col=0)
clone_res.index = [str(i) for i in clone_res.index]
ct_res.index = [str(i) for i in ct_res.index]
# %%
xenium.obs = xenium.obs.join(clone_res).join(ct_res)
# %%

clones_columns = clone_res.columns
ct_columns = ct_res.columns
xenium.obs["clone"] = xenium.obs[clones_columns].idxmax(axis = 1)
xenium.obs.clone = xenium.obs.clone.astype("str")
xenium.obs["cell_type"] = xenium.obs[ct_columns].idxmax(axis = 1)
#%%
import matplotlib.colors

palette_clone = sns.color_palette("tab20", len(clones_columns))
palette_clone = dict(zip(sorted(clones_columns), palette_clone))
palette_ct = sns.color_palette("Paired", len(ct_columns))
palette_ct = dict(zip(sorted(ct_columns), palette_ct))
palette_ct = [palette_ct[k] for k in sorted(ct_columns)]
cmap_ct = matplotlib.colors.ListedColormap(palette_ct)

palette_clone = [palette_clone[k] for k in sorted(clones_columns)]
cmap_clone = matplotlib.colors.ListedColormap(palette_clone)
#%%
xenium.obs["C_scANVI_cell_type"] = xenium.obs["C_scANVI_cell_type"].astype(str).fillna("NA")
xenium.obs["C_scANVI_cell_type"] = xenium.obs["C_scANVI_cell_type"].astype(str).fillna("NA")

#%%
sq.pl.spatial_scatter(xenium, library_id="spatial", shape=None,color=
["cell_type","C_scANVI_cell_type"],
    wspace=0.4)
#%%
sq.pl.spatial_scatter(xenium, library_id="spatial", shape=None,color=
["clone","C_scANVI_clone"],
    wspace=0.4)

#%%
ct = "PVL"
xenium.obs["space_tree"] = xenium.obs["cell_type"] == ct
xenium.obs["scanvi"] = xenium.obs["C_scANVI_cell_type"] == ct
sq.pl.spatial_scatter(xenium, library_id="spatial", shape=None,color=
["space_tree","scanvi"],
    wspace=0.4)

# %%
clone_col_high = ['2','4','5','6','7','8','10','11','12','14','15','diploid']
with mpl.rc_context({'axes.facecolor':  'black',
                     'figure.figsize': [4.5, 5]}):

    sq.pl.spatial_scatter(xenium, cmap='magma',library_id="spatial", shape=None, color = clone_col_high,
                          ncols = 3, size = 1.3
                  # show first 8 cell types
                  
                 )
# %%
norm_sim = np.load("../data/interim/clone_dist_over.npy")
norm_sim = pd.DataFrame(norm_sim, index = clones_columns, columns = clones_columns)
pdist = distance.pdist(norm_sim)
link = linkage(pdist, method='complete')
colors = sns.color_palette("tab10")

t = 1.4
fig = plt.figure(figsize=(25, 6))
with plt.rc_context({'lines.linewidth': 3}):

    dn = dendrogram(link,color_threshold = t, labels = norm_sim.columns)
plt.axhline(t, ls = "--", color = "grey", lw = 2 )
# plt.savefig("../reports/figures/dendrogram.pdf", dpi = 300)

plt.show()
clusters = fcluster(link, t, criterion='distance')
clone_mat = xenium.obs[clones_columns]

new_clones = []
new_cols = []
un_clusters = np.unique(clusters)
xenium.obs["agg_clones_st"] = xenium.obs["clone"].copy()
xenium.obs["agg_clone_scanvy"] = xenium.obs["C_scANVI_clone"].copy()
for cl in un_clusters:
    idx = np.where(clusters == cl)[0]
    old_clones =  norm_sim.columns[idx]
    new_clone_id = " & ".join([str(x) for x in old_clones])
    for old in old_clones:
        xenium.obs["agg_clones_st"].replace(old, new_clone_id, inplace = True)
        xenium.obs["agg_clone_scanvy"].replace(old, new_clone_id, inplace = True)

# %%
sq.pl.spatial_scatter(xenium, library_id="spatial", shape=None,color=
["agg_clones_st","agg_clone_scanvy"],
    wspace=0.4)
#%%
# %%
palette_clone = {'0 & 13 & 16': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
 '1 & 15': (1.0, 0.4980392156862745, 0.054901960784313725),
 '10': (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
 '12 & 14': (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
 '2 & 4 & 11 & diploid': (0.5803921568627451,
  0.403921568627451,
  0.7411764705882353),
 '5 & 8': (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
 '6 & 7 & 9': (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
 'nan': (0.4980392156862745, 0.4980392156862745, 0.4980392156862745)}



plt.figure(figsize=(5, 7))
df = xenium.obs[["x_centroid", "y_centroid", "agg_clone_scanvy"]].copy()
df["x_centroid"] = -df["x_centroid"] 
df["y_centroid"] = -df["y_centroid"] 

sns.scatterplot(data = df, x = "y_centroid", y = "x_centroid", hue = "agg_clone_scanvy",
                 alpha = 1, s = 1, palette = palette_clone)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.axis("off")
plt.show()
                 )
# %%
ct_columns = list(set(xenium.obs.cell_type))
palette_ct = sns.color_palette("Paired", len(ct_columns))
palette_ct = dict(zip(ct_columns, palette_ct))
palette_ct['nan'] = (0.4980392156862745, 0.4980392156862745, 0.4980392156862745)
# %%
plt.figure(figsize=(5, 7))
df = xenium.obs[["x_centroid", "y_centroid", "cell_type"]].copy()
df["x_centroid"] = -df["x_centroid"] 
df["y_centroid"] = -df["y_centroid"] 

sns.scatterplot(data = df, x = "y_centroid", y = "x_centroid", hue = "cell_type",
                 alpha = 1, s = 1, palette = palette_ct)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.axis("off")
plt.show()


# %%
plt.figure(figsize=(5, 7))
df = xenium.obs[["x_centroid", "y_centroid", "C_scANVI_cell_type"]].copy()
df["x_centroid"] = -df["x_centroid"] 
df["y_centroid"] = -df["y_centroid"] 

sns.scatterplot(data = df, x = "y_centroid", y = "x_centroid", hue = "C_scANVI_cell_type",
                 alpha = 1, s = 1, palette = palette_ct)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.axis("off")
plt.show()

# %%
