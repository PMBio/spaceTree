#%%
import scanpy as sc
import anndata as ad
# import squidpy as sq

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
import umap
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage, distance, fcluster
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
from sklearn.decomposition import PCA
import spaceTree.utils as utils
import os
import matplotlib as mpl
from spaceTree.plot_spatial import plot_spatial
os.chdir("/home/o313a/clonal_GNN/")
from sklearn.decomposition import FactorAnalysis
from spaceTree import plotting

#%%
visium = sc.read_h5ad("data/interim/visium_annotated.h5ad")
xenium1 = sc.read_h5ad("data/interim/xenium_samp1.h5ad")
xenium2 = sc.read_h5ad("data/interim/xenium_samp2.h5ad")

#%%
clone_cols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
       '10', '11', '12', '13','diploid']
ct_cols = ['DCIS 2', 'Prolif Invasive Tumor', 'Stromal', 'DCIS 1', 'Macrophages 1',
       'Invasive Tumor', 'T Cell & Tumor Hybrid', 'Myoepi KRT15+',
       'Myoepi ACTA2+', 'Stromal & T Cell Hybrid', 'CD8+ T Cells',
       'Endothelial', 'Perivascular-Like', 'B Cells', 'IRF7+ DCs',
       'CD4+ T Cells']
#%%
clones_present = list(set(visium.obs.clone).union(set(xenium1.obs.clone)))
clone_palette = sns.color_palette("tab20", len(clones_present))
np.random.shuffle(clone_palette)
clone_lut = dict(zip(map(str, clones_present), clone_palette))
#%%
with mpl.rc_context({'axes.facecolor':  'black',
                     'figure.figsize': [10, 10]}):
    sc.pl.spatial(visium,
                  # show first 8 cell types
                  color=clone_cols, palette = clone_lut,
                img_key='lowres', alpha_img = 0.5,
                 )
#%%
plotting.plot_xenium(xenium1.obs.x_centroid, xenium1.obs.y_centroid,
                xenium1.obs["clone"], palette = clone_lut)
plotting.plot_xenium(xenium2.obs.x_centroid, xenium2.obs.y_centroid,
                xenium2.obs["clone"], palette = clone_lut)
#%%
sc.pl.spatial(visium,
                # show first 8 cell types
                color="clone", palette = clone_lut,
            img_key='lowres', alpha_img = 0.5,
                )
#%%
ct_present = list(set(visium.obs.cell_type).union(set(xenium1.obs.cell_type)).union(set(xenium1.obs.Annotation)))
ct_palette = sns.color_palette("tab20", len(ct_present))
np.random.shuffle(ct_palette
)
ct_lut = dict(zip(map(str, ct_present), ct_palette))
#%%
with mpl.rc_context({'axes.facecolor':  'black',
                     'figure.figsize': [10, 10],
                     "font.size" : 25
                     }):
    sc.pl.spatial(visium,
                  # show first 8 cell types
                  color=ct_cols, palette = ct_lut,
                img_key='lowres', alpha_img = 0
                 )
    plt.show()

#%%
plotting.plot_xenium(xenium1.obs.x_centroid, xenium1.obs.y_centroid,
                xenium1.obs["cell_type"], palette = ct_lut)
plotting.plot_xenium(xenium2.obs.x_centroid, xenium2.obs.y_centroid,
                xenium2.obs["cell_type"], palette = ct_lut)
#%%
utils.plot_xenium(xenium1.obs.x_centroid, xenium1.obs.y_centroid,
                xenium1.obs["Annotation"], palette = ct_lut)
#%%
xenium1.obs["disagreement"] = xenium1.obs["cell_type"].astype(str) != xenium1.obs["Annotation"].astype(str)

utils.plot_xenium(xenium1.obs.x_centroid, xenium1.obs.y_centroid,
                xenium1.obs["disagreement"])
#%%
xenium1.obsm["clone_fractions"] = np.log1p(xenium1.obs[clone_cols].fillna(0).values)
sc.pp.neighbors(xenium1, use_rep='clone_fractions',
                n_neighbors = 30)

# Cluster spots into regions using scanpy
sc.tl.leiden(xenium1, resolution=0.01)

# add region as categorical variable
xenium1.obs["clone_niches"] = xenium1.obs["leiden"].astype("category")
plotting.plot_xenium(xenium1.obs.x_centroid, xenium1.obs.y_centroid,
                xenium1.obs["clone_niches"])

#%%
ct_int = list(set(visium.obs.cell_type).intersection(set(xenium1.obs.cell_type)))

df = xenium1.obs[["cell_type","clone"]]
grouped = df.groupby('cell_type')['clone'].value_counts(normalize=True).unstack().fillna(0)
grouped = grouped.loc[ct_int]
# Plot
grouped.plot(kind='bar', stacked=True, figsize=(10,7), color = clone_lut)
plt.ylabel("Proportion")
plt.show()
#%%

df = visium.obs[["cell_type","clone"]]
grouped = df.groupby('cell_type')['clone'].value_counts(normalize=True).unstack().fillna(0)
grouped = grouped.loc[ct_int]
# Plot
grouped.plot(kind='bar', stacked=True, figsize=(10,7), color = clone_lut)
plt.ylabel("Proportion")
plt.show()
#%%
norm_sim = np.load("data/interim/clone_dist_over.npy")
norm_sim = pd.DataFrame(norm_sim, index = clone_cols, columns = clone_cols)
pdist = distance.pdist(norm_sim)
link = linkage(pdist, method='complete')
colors = sns.color_palette("tab10")

t = 0.5
fig = plt.figure(figsize=(25, 6))
with plt.rc_context({'lines.linewidth': 3}):

    dn = dendrogram(link,color_threshold = t, labels = norm_sim.columns)
plt.axhline(t, ls = "--", color = "grey", lw = 2 )
# plt.savefig("../reports/figures/dendrogram.pdf", dpi = 300)

plt.show()

#%%
clusters = fcluster(link, t, criterion='distance')
clone_mat_xen1 = xenium1.obs[clone_cols]
clone_mat_xen2 = xenium2.obs[clone_cols]

clone_mat_vis = visium.obs[clone_cols]


new_clones = []
un_clusters = np.unique(clusters)
for cl in un_clusters:
    idx = np.where(clusters == cl)[0]
    old_clones =  norm_sim.columns[idx]
    new_clone_id = " & ".join([str(x) for x in old_clones])
    new_clone_xen1 = np.zeros(clone_mat_xen1.shape[0])
    new_clone_xen2 = np.zeros(clone_mat_xen2.shape[0])
    new_clone_vis = np.zeros(clone_mat_vis.shape[0])
    for old in old_clones:
        new_clone_xen1 = new_clone_xen1 + clone_mat_xen1[old].values
        new_clone_xen2 = new_clone_xen2 + clone_mat_xen2[old].values
        new_clone_vis = new_clone_vis + clone_mat_vis[old].values
    xenium1.obs[new_clone_id] = new_clone_xen1
    xenium2.obs[new_clone_id] = new_clone_xen2

    visium.obs[new_clone_id] = new_clone_vis
    new_clones.append(new_clone_id)

#%%
xenium1.obs["clone_agg"] = xenium1.obs[new_clones].idxmax(axis = 1)
xenium2.obs["clone_agg"] = xenium2.obs[new_clones].idxmax(axis = 1)

visium.obs["clone_agg"] = visium.obs[new_clones].idxmax(axis = 1)
#%%
new_clones = list(set(xenium1.obs.clone_agg))
agg_palette = sns.color_palette("tab10", len(new_clones))
np.random.shuffle(agg_palette)
agg_lut = dict(zip(map(str, new_clones), agg_palette))
#%%
with mpl.rc_context({'axes.facecolor':  'black',
                     'figure.figsize': [10, 10],"font.size" : 25}):
    sc.pl.spatial(visium,
                  # show first 8 cell types
                  color=new_clones, 
                img_key='lowres', alpha_img = 0.5,
                 )
#%%
utils.plot_xenium(xenium1.obs.x_centroid, xenium1.obs.y_centroid,
                xenium1.obs["clone_agg"], palette = agg_lut)
#%%
utils.plot_xenium(xenium2.obs.x_centroid, xenium2.obs.y_centroid,
                xenium2.obs["clone_agg"], palette = agg_lut)

#%%

vis_clone_mat = np.log1p(visium.obs[clone_cols])
vis_ct_mat = visium.obs[ct_cols]
xen_clone_mat1 = np.log1p(xenium1.obs[clone_cols].fillna(0))
xen_ct_mat1 = xenium1.obs[ct_cols].fillna(0)

xen_clone_mat2 = np.log1p(xenium2.obs[clone_cols].fillna(0))
xen_ct_mat2 = xenium2.obs[ct_cols].fillna(0)


#%%
reducer = FactorAnalysis(n_components=10, random_state=0)   
embedding = reducer.fit_transform(pd.concat([vis_clone_mat,xen_clone_mat1,xen_clone_mat2]))
#%%
source = np.repeat("visium", vis_clone_mat.shape[0])
target1 = np.repeat("xenium_sample1", xen_clone_mat1.shape[0])
target2 = np.repeat("xenium_sample2", xen_clone_mat2.shape[0])

source_target = np.concatenate([source,target1,target2])
embedding = pd.DataFrame(embedding, columns = [f"factor_{i}" for i in range(embedding.shape[1])])

embedding.index = list(visium.obs.index) + list(xenium1.obs.index) + list(xenium2.obs.index)
#%%
embedding["source"] = source_target
embedding["clone"] = pd.concat([visium.obs["clone"],xenium1.obs["clone"],xenium2.obs["clone"]]).values
#%%
grouped = embedding.groupby(["source","clone"])[[f"factor_{i}" for i in range(10)]].mean().reset_index()
#%%
grouped.index = [f"{tup[1]}:clone{tup[2]}"  for tup in grouped.itertuples()]
grouped = grouped[[f"factor_{i}" for i in range(10)]]
grouped = grouped.astype(float)

#%%
row_lables1 = [x.split(":")[0] for x in grouped.index]
row_lut1 = dict(zip(set(row_lables1), sns.color_palette("Set2", len(set(row_lables1)))))
row_labels2 = [x.split(":")[1] for x in grouped.index]
row_lut2 = dict(zip(set(row_labels2), sns.color_palette("tab20", len(set(row_labels2)))))
row_colors1 = pd.Series(row_lables1, index=grouped.index).map(row_lut1)
row_colors2 = pd.Series(row_labels2, index=grouped.index).map(row_lut2)

#%%
g = sns.clustermap(grouped,
               cmap="Blues", figsize=(10,10), cbar_kws={"label": "Factor loading"},
               xticklabels=1, yticklabels=1, row_colors=[row_colors1,row_colors2])

for label in row_lut1:
    g.ax_col_dendrogram.bar(0, 0, color=row_lut1[str(label)], label=label, linewidth=0)
l1 = g.ax_col_dendrogram.legend(title='Source', loc="upper right", ncol=1, bbox_to_anchor=(1.2,0.55), bbox_transform=plt.gcf().transFigure)

for label in row_lut2:
    g.ax_row_dendrogram.bar(0, 0, color=row_lut2[label], label=label, linewidth=0)
l2 = g.ax_row_dendrogram.legend(title='Clone', loc='upper right', ncol=2, bbox_to_anchor=(1.8, 3.5))
#%%

#%%

#%%
sns.scatterplot(x="factor2", y="factor3", hue = "source",
                                 data = embedding.loc[embedding.index[::-1]])
plt.show()

#%%
ax = sns.scatterplot(x="factor2", y="factor3",
                 hue = "clone", style = "source",
                 data = embedding.loc[embedding.index[::-1]])
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

plt.show()

#%%

idx = list(visium.obs.index) + list(xenium1.obs.index)+ list(xenium2.obs.index)
embedding.index = idx
# %%
cols = ["factor1","factor2", "factor3"]

visium.obs = visium.obs.join(embedding[cols])
xenium1.obs = xenium1.obs.join(embedding[cols])
xenium2.obs = xenium2.obs.join(embedding[cols])

# %%
plot_spatial(
    adata=visium,
    # labels to show on a plot
    color=cols, labels=cols,
    show_img=False,
    # 'fast' (white background) or 'dark_background'
    style='fast',
    # limit color scale at 99.2% quantile of cell abundance
    max_color_quantile=0.992,
    # size of locations (adjust depending on figure size)
    circle_diameter=2.5,
    colorbar_position='right',
    reorder_cmap = [0,2,1]
)
# %%
xenium1.obsm["spatial"] = utils.rotate_90_degrees_clockwise(xenium1.obsm["spatial"])

#%%
plot_spatial(
    adata=xenium1,
    # labels to show on a plot
    color=cols, labels=cols,
    show_img=False,
    # 'fast' (white background) or 'dark_background'
    style='fast',
    # limit color scale at 99.2% quantile of cell abundance
    max_color_quantile=0.992,
    # size of locations (adjust depending on figure size)
    circle_diameter=1,
    colorbar_position='right',
    reorder_cmap = [0,2,1]
)
#%%
xenium2.obsm["spatial"] = utils.rotate_90_degrees_clockwise(xenium2.obsm["spatial"])

# %%
plot_spatial(
    adata=xenium2,
    # labels to show on a plot
    color=cols, labels=cols,
    show_img=False,
    # 'fast' (white background) or 'dark_background'
    style='fast',
    # limit color scale at 99.2% quantile of cell abundance
    max_color_quantile=0.992,
    # size of locations (adjust depending on figure size)
    circle_diameter=1,
    colorbar_position='right',
    reorder_cmap = [0,2,1]
)
# %%
#%%
with mpl.rc_context({'axes.facecolor':  'black',
                     'figure.figsize': [10, 10],"font.size" : 25}):
    sc.pl.spatial(visium,
                  # show first 8 cell types
                  color=['1 & 11', 'diploid', '5', '15', '8', '3', '25'], 
                img_key='lowres', alpha_img = 0.5,
                 )
# %%
