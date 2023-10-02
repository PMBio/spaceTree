import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
def get_results(pred, data, node_encoder_rev, node_encoder_ct):
    pred_clone = pred[:,:data.num_classes_clone-1]
    pred_cell_type = pred[:,data.num_classes_clone-1:]
    pred_clone = np.exp(F.log_softmax(pred_clone, dim=1).detach().cpu().numpy())
    pred_cell_type = np.exp(F.log_softmax(pred_cell_type, dim=1).detach().cpu().numpy())
    cells_hold_out =[node_encoder_rev[x.item()] for x in data.hold_out]
    clone_res = pd.DataFrame(pred_clone[data.hold_out.detach().cpu().numpy()], index = cells_hold_out)
    clone_res.columns =list(np.arange(pred_clone.shape[1]))[:-1] + ["diploid"]
    ct_res = pd.DataFrame(pred_cell_type[data.hold_out.detach().cpu().numpy()], index = cells_hold_out)
    ct_res.columns = [node_encoder_ct[x] for x in ct_res.columns]
    return(clone_res,ct_res)

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

def get_attention_visium(w,node_encoder_rev, data,coordinates):
    edges = w[0]
    weight = w[1]
    edge, edge_weight = remove_self_loops(edges, weight)
    spatial_nodes = set(list(data.hold_out.cpu().numpy()))
    spatial_graph = {}
    for i in tqdm(range(edge.shape[1])):
        source = edge[0][i].item()
        target = edge[1][i].item()
        source_id = node_encoder_rev[source]
        target_id = node_encoder_rev[target]
        if target in spatial_nodes:
            if target_id not in spatial_graph:
                spatial_graph[target_id] = []
            if  source == target:
                spatial_graph[target_id].append((source_id,edge_weight[i].item(), "self"))
                
            elif source in spatial_nodes:
                spatial_graph[target_id].append((source_id,edge_weight[i].item(), "spatial"))
            else:
                spatial_graph[target_id].append((source_id,edge_weight[i].item(), "reference"))
    full_df = []
    for key in tqdm(spatial_graph):
        tmp = pd.DataFrame(spatial_graph[key], columns = ["source", "weight", "type"])
        tmp.drop_duplicates(inplace=True)
        tmp["target"] = key
        full_df.append(tmp)
    full_df = pd.concat(full_df)
    ds = []
    for tup in tqdm(full_df.itertuples()):
        if tup.type == "spatial":
            source = str(tup.source)
            target = str(tup.target)
            if source in coordinates.index and target in coordinates.index:
                source_coor = coordinates.loc[source].values
                target_coor = coordinates.loc[target].values
                dist = np.sum(np.abs(source_coor - target_coor))
                if dist == 2:
                    dist = "first_neighbour"
                elif dist == 4:
                    dist = "second_neighbour"
            else:
                dist = "reference"
        else:
            dist = "reference"
        ds.append(dist)
    full_df["distance"] = ds



    full_df = full_df[["target","distance","weight"]].groupby(["target", "distance"]).sum().reset_index()
    full_df = full_df.set_index("target")
    sns.histplot(full_df, x = "weight", hue = "distance", bins = 50, log_scale = False)
    plt.show()

    full_df = full_df.pivot(columns = "distance", values = "weight")

    return(full_df)

def plot_xenium(x,y,hue,palette = None):
    x= -x 
    y = -y
    if isinstance(hue, pd.Series):
        n_plots = 1
    else:
        n_plots = hue.shape[1]
    fig, axs = plt.subplots(1, n_plots, figsize=(5* n_plots, 7))
    if n_plots != 1:
        for i in range(n_plots):
            ax = axs[i]
            sns.scatterplot(x = y, y = x, hue = hue.iloc[:,i],
                            alpha = 1, s = 1, palette = palette, ax = ax)
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            ax.axis("off")
    else:
        ax = axs
        sns.scatterplot(x = y, y = x, hue = hue,
                        alpha = 1, s = 1, palette = palette, ax = ax)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        ax.axis("off")
    plt.show()


def get_attention(w,node_encoder_rev, data,coordinates):
    edges = w[0]
    weight = w[1]
    edge, edge_weight = remove_self_loops(edges, weight)
    spatial_nodes = set(list(data.hold_out.cpu().numpy()))
    spatial_graph = {}
    for i in tqdm(range(edge.shape[1])):
        source = edge[0][i].item()
        target = edge[1][i].item()
        source_id = node_encoder_rev[source]
        target_id = node_encoder_rev[target]
        if target in spatial_nodes:
            if target_id not in spatial_graph:
                spatial_graph[target_id] = []
            if  source == target:
                spatial_graph[target_id].append((source_id,edge_weight[i].item(), "self"))
                
            elif source in spatial_nodes:
                spatial_graph[target_id].append((source_id,edge_weight[i].item(), "spatial"))
            else:
                spatial_graph[target_id].append((source_id,edge_weight[i].item(), "reference"))
    full_df = []
    for key in tqdm(spatial_graph):
        tmp = pd.DataFrame(spatial_graph[key], columns = ["source", "weight", "type"])
        tmp.drop_duplicates(inplace=True)
        tmp["target"] = key
        full_df.append(tmp)
    full_df = pd.concat(full_df)
    ds = []
    for tup in tqdm(full_df.itertuples()):
        if tup.type == "spatial":
            source = str(tup.source)
            target = str(tup.target)
            if source in coordinates.index and target in coordinates.index:
                source_coor = coordinates.loc[source].values
                target_coor = coordinates.loc[target].values
                dist = np.sum(np.abs(source_coor - target_coor))
            else:
                dist = 0
        else:
            dist = 0
        ds.append(dist)
    full_df["distance"] = ds
    spatial = full_df[full_df.type != "reference"]
    sc = full_df[full_df.type == "reference"]
    spatial["distance"] = pd.cut(spatial.distance,3, labels = ["short","medium","long"])
    sc["distance"] = "reference"
    full_df = pd.concat([sc,spatial])
    full_df = full_df[["target","distance","weight"]].groupby(["target", "distance"]).sum().reset_index()
    sns.histplot(full_df, x = "weight", hue = "distance", bins = 50, log_scale = False)
    plt.show()

    full_df = full_df.pivot(columns = "distance", values = "weight", index = "target")
    return(full_df)


