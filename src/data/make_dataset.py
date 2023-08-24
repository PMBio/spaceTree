# -*- coding: utf-8 -*-
import torch
import pandas as pd
from torch_geometric.data import Data
import pickle   

def process_data():
    """
    Processes the raw data and saves the resulting tensor to a specified location.
    """
    # Read the required data files
    edges = pd.read_csv("../../data/interim/edges.csv")
    overcl = pd.read_csv("../../data/interim/clones_over.csv")
    emb_vis = pd.read_csv("../../data/interim/embedding_visium_scvi.csv", index_col=0)
    emb_rna = pd.read_csv("../../data/interim/embedding_rna2vis_scvi.csv", index_col=0)

    # Clean and process the data
    overcl.columns = ["node1", "clone"]
    edges = edges.drop(columns=["clone"])
    
    # Merging the edges with overcl data
    overcl = edges[edges["type"] == "sc2vis"].merge(overcl, on="node1", how="left")
    overcl.clone = overcl.clone.fillna("diploid")
    
    # Merging processed overcl data with original edges
    edges = edges.merge(overcl[["clone", "node1"]], on="node1", how="left")

    # Filtering the nodes that are present in both graph and embeddings
    all_nodes_graph = set(edges.node1.to_list() + edges.node2.to_list())
    all_nodes_emb = set(emb_vis.index).union(set(emb_rna.index))
    all_nodes = list(all_nodes_graph.intersection(all_nodes_emb))
    node_encoder = {all_nodes[i]: i for i in range(len(all_nodes))}
    
    # Filtering embeddings and edges with valid nodes
    emb_vis = emb_vis.loc[emb_vis.index.isin(all_nodes)]
    emb_rna = emb_rna.loc[emb_rna.index.isin(all_nodes)]
    edges = edges[edges.node1.isin(all_nodes) & edges.node2.isin(all_nodes)]
    
    # Encoding nodes
    edges.node1 = edges.node1.map(node_encoder)
    edges.node2 = edges.node2.map(node_encoder)
    emb_vis = emb_vis.rename(index=node_encoder)
    emb_rna = emb_rna.rename(index=node_encoder)

    # Building edge_index and features for PyTorch Geometric Data
    edge_index = torch.tensor([edges.node1, edges.node2], dtype=torch.long)
    features = pd.concat([emb_vis, emb_rna]).sort_index()
    x = torch.tensor(features.values, dtype=torch.float)
    
    # Handling missing data and mapping categories
    edges.clone = edges.clone.fillna("missing")
    nodes_atr = edges[["node1", "type", "celltype_major", "clone"]].drop_duplicates().sort_values(by="node1")
    nodes_atr.celltype_major = nodes_atr.celltype_major.fillna("missing")
    ct_list = list(nodes_atr.celltype_major.unique())
    ct_list.remove("missing")
    clone_list = list(nodes_atr.clone.unique())
    clone_list = [cl for cl in clone_list if cl not in ["missing", "diploid"]]
    clone_dict = {x: int(x) for x in clone_list}
    clone_dict.update({"missing": -1, "diploid": len(clone_dict) - 1})
    type_dict = {ct_list[i]: i for i in range(len(ct_list))}
    
    # Encoding categories
    nodes_atr.clone = nodes_atr.clone.map(clone_dict)
    nodes_atr.celltype_major = nodes_atr.celltype_major.map(type_dict)
    nodes_atr = nodes_atr.set_index("node1")
    features = features.join(nodes_atr).fillna(-1)
    
    # Creating tensors from processed data
    y_clone = torch.tensor(features.clone.values, dtype=torch.long)
    y_type = torch.tensor(features.celltype_major.values, dtype=torch.long)
    
    # Building the Data object for PyTorch Geometric
    data = Data(x=x, edge_index=edge_index, y_clone=y_clone, y_type=y_type, edge_type=edges.type.values)

    # Ensure the data is valid
    assert data.validate(raise_on_error=True)

    # Save the processed data
    torch.save(data, "../../data/processed/data.pt")
    node_encoder.update(clone_dict)
    node_encoder.update(type_dict)
    with open('../../data/processed/node_encoding.pkl', 'wb') as fp:
        pickle.dump(node_encoder, fp)
if __name__ == '__main__':
    process_data()