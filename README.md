<div style="text-align: center;">
  <img src="https://github.com/PMBio/spaceTree/blob/master/docs/space_tree.png?raw=true" alt="spacetree logo" width="200"/>
</div>

spaceTree: Deciphering Tumor Microenvironments by joint modeling of cell states and genotype-phenotype relationships in spatial omics data 
==============================

spaceTree jointly models spatially smooth cell type- and clonal state composition.
spaceTree employs Graph Attention mechanisms, capturing information from spatially close regions when reference mapping falls short, enhancing both interpretation and quantitative accuracy. 

A significant merit of spaceTree is its technology-agnostic nature, allowing clone-mapping in sequencing- and imaging-based assays. 
The model outputs can be used to characterize spatial niches that have consistent cell type and clone composition.


<div style="text-align: center;">
  <img src="https://github.com/PMBio/spaceTree/blob/master/docs/schema.jpg?raw=true" alt="spacetree schema" width="1000"/>
</div>

Overview of the spatial mapping approach and the workflow enabled by spaceTree.From left to right: spaceTree requirs as input reference (scRNA-seq) and spatial count matrices as well as labels that need to be transfered. The labels can be descrete, continious or hierachical. The model outputs a spatial mapping of the labels and the cell type (compositions in case of Visium) of the spatial regions.

# Usage and Tutorials


## Installation

### pytorch & pytorch geometric dependencies
SpaceTree reles on `pytorch`,`pytorch geometric` and `pyg-lib` libraries for GNNs and efficient graph sampling routines. It was develoed and tested with `pytorch==2.0.1`, `torch-geometric==2.5.0` and `pyg-lib==0.2.0+pt20cu118`. We recommend to use the same versions, when possible, otherwise just go with the ones that are compatable with your CUDA version. 

To install versions compatible with your CUDA version, please visit the offical documentation of [pytorch](https://pytorch.org/get-started/locally/) (1), [pytorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) (2) and [pyg-lib](https://github.com/pyg-team/pyg-lib) (3) and complete the installations **in that order**.

Please note, that access to GPU is adviced, but not nessesary, especially if the data size is not too large (i.e. for Visium HD we strongly recommend to use GPU).

#### Example installation routine

To demonstrate the logic, here is **an example** installation for MacOS 14 without CUDA (CPU-only) and Python 3.10 (if that is not your desired configuration, please **do not** adjust the commands yourself, but refer to the official documentation of the libraries, because syntax is platform dependent and some versions might be not compatable with each other):
```python
conda create -y -n spacetree_env python=3.10
conda activate spacetree_env
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 -c pytorch
pip install torch_geometric
pip install pyg_lib -f https://data.pyg.org/whl/torch-2.3.0+cpu.html 
#test the installation
python -c "import torch_geometric; print(torch_geometric.typing.WITH_PYG_LIB)"
#TRUE
```
If the output is `TRUE`, then the installation was successful. If not, please check the error message and try to resolve the issue based on the pytorch, pytorch geometric and pyg-lib documentation.

### spaceTree Installation
Once you completed the installation of the dependencies, you can install spaceTreeusing pip or from source.

Installation with pip:
```bash
conda activate spacetree_env
pip install spaceTree

```
Installation from source:
```bash
conda activate spacetree_env
git clone https://github.com/PMBio/spaceTree.git
# cd in the spaceTree directory
cd spaceTree
pip install .

```

## Documentation, Tutorials and Examples
Check out our tutorials and documentation to get started with spaceTree [here](https://pmbio.github.io/spaceTree/).

## Citation
