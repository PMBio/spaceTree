import torch
import pandas as pd

from torch_geometric.data import Data

import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from collections import Counter
import pickle   
def validate_counts(counter, threshold, label):
    for key, value in counter.items():
        assert value > threshold, f"{label} {key} has less than {threshold + 1} items"
        

def filter_and_encode(df, node_encoder, all_nodes,use_index=False):
    if use_index:
        df = df[df.index.isin(all_nodes)]
        df = df.rename(index=node_encoder)
    else:
        assert len(set(df.node2).difference(set(df.node1))) == 0, "Not all nodes are present in the graph"
        df.node1 = df.node1.map(node_encoder)
        df.node2 = df.node2.map(node_encoder)

    return df

def drop_small(edges,numb):
    cl_dict = Counter(edges[["node1","clone"]].drop_duplicates().clone)
    clones_exc = [key for key in cl_dict if cl_dict[key]<=numb]
    print(f"Excluding {len(clones_exc)} clones with less than {numb} cells")
    ct_dict = Counter(edges[["node1","cell_type"]].drop_duplicates().cell_type)
    ct_exc = [key for key in ct_dict if ct_dict[key]<=numb]
    print(f"Excluding {len(ct_exc)} cell types with less than {numb} cells")
    edges = edges[~edges.clone.isin(clones_exc)]
    edges = edges[~edges.cell_type.isin(ct_exc)]
    edges = edges[~edges.node2.isin(set(edges.node2).difference(set(edges.node1)))]
    return(edges)

def preprocess_data(edges, overcl, spatial_edges, grid_edges):
    # Filter and fill missing values
    overcl = overcl.merge(edges[edges["type"] != spatial_edges], on="node1", how="left")
    overcl = overcl[["node1", "clone", "cell_type"]].drop_duplicates()    
    edges = edges.merge(overcl[["clone", "node1", "cell_type"]], on="node1", how="left")
    drop_nodes_clone = edges[(edges["type"]!= grid_edges)&(edges.clone.isna())].node1
    drop_nodes_ct = edges[(edges["type"]!=grid_edges)&(edges.cell_type.isna())].node1
    to_drop = list(set(drop_nodes_clone).union(set(drop_nodes_ct)))
    edges = edges[~edges.node1.isin(to_drop)]
    edges = edges[~edges.node2.isin(to_drop)]
    return edges, overcl


def read_and_merge_embeddings(paths, edges, drop_less = 10):
    all_nodes_graph = set(edges.node1).union(set(edges.node2))
    emb_vis = pd.read_csv(paths["spatial"], index_col=0)
    emb_vis.index = emb_vis.index.map(str)
    emb_rna = pd.read_csv(paths["rna"], index_col=0)
    emb_rna.index = emb_rna.index.map(str)

    all_nodes_emb = set(emb_vis.index).union(set(emb_rna.index))
    all_nodes = list(all_nodes_graph.intersection(all_nodes_emb))
    edges = edges[edges.node1.isin(all_nodes) & edges.node2.isin(all_nodes)]
    edges = drop_small(edges,drop_less)
    all_nodes_graph = set(edges.node1).union(set(edges.node2))
    all_nodes = list(all_nodes_graph.intersection(all_nodes_emb))

    node_encoder = {all_nodes[i]:i for i in range(len(all_nodes))}
    emb_vis = filter_and_encode(emb_vis, node_encoder, all_nodes, use_index=True)
    emb_rna = filter_and_encode(emb_rna, node_encoder, all_nodes, use_index=True)
    edges = filter_and_encode(edges, node_encoder, all_nodes)

    
    return emb_vis, emb_rna, edges, node_encoder


def create_data_object(edges, emb_vis, emb_rna,node_encoder, sim = None, with_diploid = True):
    # Convert to tensors
    edge_index = torch.tensor([edges.node1.values, edges.node2.values], dtype=torch.long)
    edge_weight = torch.tensor(edges.weight.values, dtype=torch.float)
    features = pd.concat([emb_vis, emb_rna]).sort_index()
    x = torch.tensor(features.values, dtype=torch.float)
    
    # Encode attributes
    edges.clone = edges.clone.fillna("missing")
    edges.cell_type = edges.cell_type.fillna("missing")

    nodes_attr = edges[["node1", "cell_type", "clone"]].drop_duplicates().sort_values(by="node1")
    if with_diploid:
        clone_dict = create_encoding_dict(nodes_attr, "clone", extras=["diploid", "missing"])
    else:
        clone_dict = create_encoding_dict(nodes_attr, "clone", extras=["missing"])
    type_dict = create_encoding_dict(nodes_attr, "cell_type", extras=["missing"])
    
    nodes_attr["clone"] = nodes_attr["clone"].map(clone_dict)
    nodes_attr["cell_type"] = nodes_attr["cell_type"].map(type_dict)
    nodes_attr = nodes_attr.set_index("node1")
    features = features.join(nodes_attr)
    
    y_clone = torch.tensor(features.clone.values, dtype=torch.long)
    y_type = torch.tensor(features.cell_type.values, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, y_clone=y_clone, y_type=y_type, edge_type=edges.type.values, edge_attr=edge_weight)
    assert data.validate(raise_on_error=True), "Data not valid"
    assert data.x.shape[0] == data.y_clone.shape[0] == data.y_type.shape[0], "Data not valid"
    
    clone_vals = list(clone_dict.keys())
    clone_vals.remove("missing")
    if sim is not None:
        sim = sim.loc[clone_vals,clone_vals]
        sim = sim.rename(index = clone_dict,columns =clone_dict)
        s = sorted(sim.columns)
        sim = sim.loc[s,s].values
        return data, {"nodes": node_encoder, "clones": clone_dict, "types": type_dict},sim
    else:
        return data, {"nodes": node_encoder, "clones": clone_dict, "types": type_dict}



def create_encoding_dict(df, column, extras=[]):
    items = list(df[column].unique())
    for extra in extras:
        items.remove(extra)
    if column == 'clone':
        dt = {item: idx for idx, item in enumerate(items)}
        dt["missing"] = -1
        dt["diploid"] = len(dt)-1
    else:
        dt = {item: idx for idx, item in enumerate(items)}
        dt["missing"] = -1


    return dt