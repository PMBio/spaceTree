---
layout: default
title: "Welcome to spaceTree's Documentation"
nav_order: 1
---
# Welcome to spaceTree's Documentation

<div style="text-align: left;">
  <img src="space_tree.png" alt="spacetree logo" width="200"/>
</div>


Welcome to the official documentation for spaceTree! Here, you will find tutorials and guides to help you get started and make the most out of our tool.
# Table of contents
1. [Installation](#installation)
    1. [pytorch & pytorch geometric dependencies](#dependency)
    2. [spaceTree Installation](#spacetree)
2. [Tutorials](tutorials/index.md)
    1. [Cell state and clone mapping to 10x Visium with SpaceTree](tutorials/cell-state-clone-mapping.md)
    2. [Cell state and clone mapping to 10x Xenium with SpaceTree](tutorials/cell-state-clone-mapping-xenium.md)
    3. [Visium HD data tips and tricks](tutorials/visium-hd.md)
    
## Installation <a name="installation"></a>
### pytorch & pytorch geometric dependencies <a name="dependency"></a>
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

### spaceTree Installation <a name="spacetree"></a>
Once you completed the installation of the dependencies, you can install spaceTree using pip or from source.

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

## Tutorials <a name="tutorials"></a>
spaceTree is a versatile tool that can be used in a variety of applications. It works with both sequencing and imaging-based assays. The main difference in tutorials is the way spatial graph needs to be constructed. For technologies that are based on a grid (like Visium ST/HD) we rely on the grid for the graph construction. For technologies like Xenium, we compute the spatial similarity graph based on the spatial coordinates of the cells.

To understand the workflow please refer to our end-to-end tutorials.

For Visium/grid-based data:

- [Cell state and clone mapping to 10x Visium with SpaceTree](tutorials/cell-state-clone-mapping.md)

For Xenium data, please refer to:
- [Cell state and clone mapping to 10x Xenium with SpaceTree](tutorials/cell-state-clone-mapping-xenium.md)

We do not provide a separate tutorial for Visium HD data, as the workflow is the same as for Visium data. However, we provide some tips and tricks for working with Visium HD data [here](tutorials/visium-hd.md).

If you need help defining clones based on your own scRNA-seq data, you can use tools such as [inferCNV](https://github.com/broadinstitute/inferCNV/wiki), [inferCNVpy](https://infercnvpy.readthedocs.io/en/latest/tutorials.html), [copyKAT](https://github.com/navinlabcode/copykat) and others.

For the sake of the Visium and Xenium tutorials, we also show how we ran the clone inference based on inferCNVpy [here](https://github.com/PMBio/spaceTree/blob/master/notebooks/infercnv_run.ipynb)

