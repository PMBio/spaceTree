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


xenium = sc.read_h5ad("../data/interim/scanvi_xenium.h5ad")

# %%
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
xenium.obsm["spatial"] = rotate_90_degrees_clockwise(xenium.obs[["x_centroid", "y_centroid"]].copy().to_numpy())
#%%
sq.pl.spatial_scatter(xenium, library_id="spatial", shape=None,color=[
"C_scANVI_cell_type"
    ],
    wspace=0.4,)
#%%
clones_columns = [str(x) for x in range(17)] + ["diploid"]
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
clone_col = xenium.obs[SCANVI_PREDICTION_KEY].copy()

new_clones = []
new_cols = []
un_clusters = np.unique(clusters)
for cl in un_clusters:
    idx = np.where(clusters == cl)[0]
    old_clones =  norm_sim.columns[idx]
    new_id = " & ".join([str(x) for x in old_clones])
    for clone in old_clones:
        clone_col = clone_col.replace(clone, new_id)
    new_clones.append(new_id)


#%%
xenium.obs["clones_agg"] = clone_col
#%%
xenium_tmp = xenium.copy()
xenium_tmp = xenium_tmp[xenium_tmp.obs[xenium_tmp.obs.clones_agg.isin(new_clones)].index]
#%%
palette_clone = {'0 & 13 & 16': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
 '1 & 15': (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
 '10': (1.0, 0.4980392156862745, 0.054901960784313725),
 '12 & 14': (1.0, 0.7333333333333333, 0.47058823529411764),
 '2 & 4 & 11 & diploid': (0.17254901960784313,
  0.6274509803921569,
  0.17254901960784313),
 '5 & 8': (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
 '6 & 7 & 9': (0.8392156862745098, 0.15294117647058825, 0.1568627450980392)}
#%%
plt.figure(figsize=(5, 7))
df = xenium_tmp.obs[["x_centroid", "y_centroid", "clones_agg"]].copy()
df["x_centroid"] = -df["x_centroid"] 
df["y_centroid"] = -df["y_centroid"] 

sns.scatterplot(data = df, x = "y_centroid", y = "x_centroid", hue = "clones_agg", 
                alpha = 1, s = 1, palette = palette_clone)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.axis("off")
plt.show()
#%%
sq.pl.spatial_scatter(xenium, library_id="spatial", shape=None,color=[
"clones_agg"
    ],
    wspace=0.4,)
#%%

xenium.obs[SCANVI_PREDICTION_KEY] = xenium.obs[SCANVI_PREDICTION_KEY].astype(str).fillna("x")

# %%
import matplotlib as mpl

clones_columns = list(xenium.obs[SCANVI_PREDICTION_KEY].unique())
palette_clone = sns.color_palette("tab20", len(clones_columns))
palette_clone = dict(zip(sorted(clones_columns), palette_clone))
palette_clone = [palette_clone[k] for k in sorted(clones_columns)]
cmap_clone = mpl.colors.ListedColormap(palette_clone)
# %%
sq.pl.spatial_scatter(xenium, library_id="spatial", shape=None,color=[
SCANVI_PREDICTION_KEY
    ],palette = cmap_clone,
    wspace=0.4,)
# %%
