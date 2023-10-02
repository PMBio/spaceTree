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


#%%
visium = sc.read_h5ad("data/interim/visium_annotated.h5ad")
xenium = sc.read_h5ad("data/interim/xenium_annotated_sampler.h5ad")
#%%
clone_cols = ['0', '1', '2', '3', '4', '5',
       '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', 'diploid']
ct_cols = ['DCIS 2', 'Prolif Invasive Tumor', 'Stromal', 'DCIS 1', 'Macrophages 1',
       'Invasive Tumor', 'T Cell & Tumor Hybrid', 'Myoepi KRT15+',
       'Myoepi ACTA2+', 'Stromal & T Cell Hybrid', 'unknown', 'CD8+ T Cells',
       'Endothelial', 'Perivascular-Like', 'B Cells', 'IRF7+ DCs',
       'CD4+ T Cells', 'Mast Cells', 'Macrophages 2', 'LAMP3+ DCs']
#%%
clones_present = list(set(visium.obs.clone).union(set(xenium.obs.clone)))
clone_palette = sns.color_palette("hls", len(clones_present))
clone_lut = dict(zip(map(str, clones_present), clone_palette))
#%%
with mpl.rc_context({'axes.facecolor':  'black',
                     'figure.figsize': [10, 10]}):
    sc.pl.spatial(visium,
                  # show first 8 cell types
                  color="clone", palette = clone_lut,
                img_key='lowres', alpha_img = 0.5,
                 )
#%%
utils.plot_xenium(xenium.obs.x_centroid, xenium.obs.y_centroid,
                xenium.obs["clone"], palette = clone_lut)
#%%
xenium.obs.cell_type.fillna("unknown", inplace = True)
xenium.obs.cell_type = xenium.obs.cell_type.replace({'DCIS 1': 'temporary_holder'})
xenium.obs.cell_type = xenium.obs.cell_type.replace({'DCIS 2': 'DCIS 1'})
xenium.obs.cell_type = xenium.obs.cell_type.replace({'temporary_holder': 'DCIS 2'})

utils.plot_xenium(xenium.obs.x_centroid, xenium.obs.y_centroid,
                xenium.obs["cell_type"], palette= "tab20")
#%%
list1 = list(set(xenium.obs.cell_type))
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
ct_present = list(set(visium.obs.cell_type).union(set(xenium.obs.cell_type)).union(set(xenium.obs.Annotation)))
ct_palette = sns.color_palette("tab20", len(ct_present))
np.random.shuffle(ct_palette
)
ct_lut = dict(zip(map(str, ct_present), ct_palette))
#%%
visium.obs.cell_type = visium.obs.cell_type.replace({'DCIS 1': 'temporary_holder'})
visium.obs.cell_type = visium.obs.cell_type.replace({'DCIS 2': 'DCIS 1'})
visium.obs.cell_type = visium.obs.cell_type.replace({'temporary_holder': 'DCIS 2'})
#%%
visium.obs = visium.obs.rename(columns ={"DCIS 1": "DCIS 2", "DCIS 2": "DCIS 1"})
xenium.obs = xenium.obs.rename(columns ={"DCIS 1": "DCIS 2", "DCIS 2": "DCIS 1"})
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
utils.plot_xenium(xenium.obs.x_centroid, xenium.obs.y_centroid,
                xenium.obs["cell_type"], palette = ct_lut)
#%%
utils.plot_xenium(xenium.obs.x_centroid, xenium.obs.y_centroid,
                xenium.obs["Annotation"], palette = ct_lut)
#%%
xenium.obs["disagreement"] = xenium.obs["cell_type"].astype(str) != xenium.obs["Annotation"].astype(str)
#%%
utils.plot_xenium(xenium.obs.x_centroid, xenium.obs.y_centroid,
                xenium.obs["disagreement"])
#%%
ct_int = list(set(visium.obs.cell_type).intersection(set(xenium.obs.cell_type)))

df = xenium.obs[["cell_type","clone"]]
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

t = 1.4
fig = plt.figure(figsize=(25, 6))
with plt.rc_context({'lines.linewidth': 3}):

    dn = dendrogram(link,color_threshold = t, labels = norm_sim.columns)
plt.axhline(t, ls = "--", color = "grey", lw = 2 )
# plt.savefig("../reports/figures/dendrogram.pdf", dpi = 300)

plt.show()

#%%
clusters = fcluster(link, t, criterion='distance')
clone_mat_xen = xenium.obs[clone_cols]
clone_mat_vis = visium.obs[clone_cols]


new_clones = []
un_clusters = np.unique(clusters)
for cl in un_clusters:
    idx = np.where(clusters == cl)[0]
    old_clones =  norm_sim.columns[idx]
    new_clone_id = " & ".join([str(x) for x in old_clones])
    new_clone_xen = np.zeros(clone_mat_xen.shape[0])
    new_clone_vis = np.zeros(clone_mat_vis.shape[0])
    for old in old_clones:
        new_clone_xen = new_clone_xen + clone_mat_xen[old].values
        new_clone_vis = new_clone_vis + clone_mat_vis[old].values
    xenium.obs[new_clone_id] = new_clone_xen
    visium.obs[new_clone_id] = new_clone_vis
    new_clones.append(new_clone_id)

#%%
xenium.obs["clone_agg"] = xenium.obs[new_clones].idxmax(axis = 1)
visium.obs["clone_agg"] = visium.obs[new_clones].idxmax(axis = 1)
#%%
agg_palette = sns.color_palette("tab10", len(new_clones))
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
utils.plot_xenium(xenium.obs.x_centroid, xenium.obs.y_centroid,
                xenium.obs["clone_agg"], palette = agg_lut)
#%%
ct_int = list(set(visium.obs.cell_type).intersection(set(xenium.obs.cell_type)))
#%%
df = visium.obs[["cell_type","clone_agg"]]
grouped = df.groupby('cell_type')['clone_agg'].value_counts(normalize=True).unstack().fillna(0)
grouped = grouped.loc[ct_int]

# Plot
grouped.plot(kind='bar', stacked=True, figsize=(10,7), color = agg_lut)
plt.ylabel("Proportion")
plt.show()
#%%
df = xenium.obs[["cell_type","clone_agg"]]
grouped = df.groupby('cell_type')['clone_agg'].value_counts(normalize=True).unstack().fillna(0)
grouped = grouped.loc[ct_int]

# Plot
grouped.plot(kind='bar', stacked=True, figsize=(10,7), color = agg_lut)
plt.ylabel("Proportion")
plt.show()

#%%


vis_clone_mat = visium.obs[new_clones]
vis_ct_mat = visium.obs[ct_cols]
xen_clone_mat = xenium.obs[new_clones].fillna(0)
xen_ct_mat = xenium.obs[ct_cols].fillna(0)

#%%

reducer = umap.UMAP(n_components=3)    
embedding_vis = reducer.fit_transform(vis_clone_mat)
#%%

reducer = umap.UMAP(n_components=3)    
embedding_xen = reducer.fit_transform(xen_clone_mat)
#%%

sns.scatterplot(x=embedding_vis[:,0], y=embedding_vis[:,1], hue = visium.obs["clone_agg"])
plt.show()
#%%
visium.obs["umap1"] = embedding_vis[:,0]
visium.obs["umap2"] = embedding_vis[:,1]
visium.obs["umap3"] = embedding_vis[:,2]

#%%
plot_spatial(
    adata=visium,
    # labels to show on a plot
    color=["umap1","umap2","umap3"], labels=["umap1","umap2","umap3"],
    show_img=True,
    # 'fast' (white background) or 'dark_background'
    style='fast',
    # limit color scale at 99.2% quantile of cell abundance
    max_color_quantile=0.992,
    # size of locations (adjust depending on figure size)
    circle_diameter=1,
    colorbar_position='right',
    reorder_cmap = [1,2,3]
)
#%%

sns.scatterplot(x=embedding_xen[:,0], y=embedding_xen[:,1], hue = xenium.obs["clone_agg"])
plt.show()
#%%
xenium.obs["umap1"] = embedding_xen[:,0]
xenium.obs["umap2"] = embedding_xen[:,1]
xenium.obs["umap3"] = embedding_xen[:,2]

#%%
plot_spatial(
    adata=xenium,
    # labels to show on a plot
    color=["umap1","umap2","umap3"], labels=["umap1","umap2","umap3"],
    show_img=False,
    # 'fast' (white background) or 'dark_background'
    style='dark_background',
    # limit color scale at 99.2% quantile of cell abundance
    max_color_quantile=0.992,
    # size of locations (adjust depending on figure size)
    circle_diameter=1,
    colorbar_position='right',
    reorder_cmap = [1,2,3]
)
#%%
reducer = umap.UMAP(n_components=3)    
embedding = reducer.fit_transform(pd.concat([vis_clone_mat,xen_clone_mat]))
#%%
source = np.repeat("visium", embedding_vis.shape[0])
target = np.repeat("xenium", embedding_xen.shape[0])
source_target = np.concatenate([source,target])
embedding = pd.DataFrame(embedding, columns = ["UMAP_1","UMAP_2", "UMAP_3"])
embedding["source"] = source_target
embedding["clone"] = pd.concat([visium.obs["clone_agg"],xenium.obs["clone_agg"]]).values
#%%
sns.scatterplot(x="UMAP_1", y="UMAP_2", hue = "source",
                                 data = embedding.loc[embedding.index[::-1]])
plt.show()

#%%
ax = sns.scatterplot(x="UMAP_2", y="UMAP_3",
                 hue = "clone", style = "source",
                 data = embedding.loc[embedding.index[::-1]])
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

plt.show()

#%%

idx = list(visium.obs.index) + list(xenium.obs.index)
embedding.index = idx
# %%
visium.obs = visium.obs.join(embedding[["UMAP_1","UMAP_2","UMAP_3"]])
xenium.obs = xenium.obs.join(embedding[["UMAP_1","UMAP_2","UMAP_3"]])

# %%
plot_spatial(
    adata=visium,
    # labels to show on a plot
    color=["UMAP_2","UMAP_3"], labels=["UMAP_2","UMAP_3"],
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
xenium.obsm["spatial"] = utils.rotate_90_degrees_clockwise(xenium.obsm["spatial"])
#%%
plot_spatial(
    adata=xenium,
    # labels to show on a plot
    color=["UMAP_2","UMAP_3"], labels=["UMAP_2","UMAP_3"],
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
