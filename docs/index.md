---
layout: default
title: "Welcome to spaceTree's Documentation"
nav_order: 1
---
# Welcome to spaceTree's Documentation

<div style="text-align: left;">
  <img src="space_tree.png" alt="spacetree logo" width="200"/>
</div>


Welcome to the official documentation for SpaceTree! Here, you will find tutorials and guides to help you get started and make the most out of our tool.

### Installation
SpaceTree reles on `pytorch geometric` and `pyg-lib` libraries for GNNs and efficient graph sampling routines. It was develoed and tested with `torch-geometric==2.5.0` and `pyg-lib==0.2.0+pt20cu118`. We recommend to use the same versions, when possible, otherwise just go with the ones that are compatable with your CUDA version. Please note, that access to GPU is adviced, but not nessesary, especially if the data size is not too large (i.e. for Visium HD we strongly recommend to use GPU).
Please visit the [offical documentation](https://github.com/pyg-team/pyg-lib) to make sure that you will install the version that is compatable with your GPUs.

Installation with pip:
```bash
conda create -y -n spacetree_env python=3.10
conda activate spacetree_env
pip install spaceTree
#install torch geometric (check the documentation for the supported versions)
pip install torch-geometric
# install pyg-lib (check supported wheels for your CUDA version)
pip install pyg_lib 
```
Installation from source:
```bash
conda create -y -n spacetree_env python=3.10
conda activate spacetree_env
git clone https://github.com/PMBio/spaceTree.git
# cd in the spaceTree directory
cd spaceTree
pip install .
#install torch geometric (check the documentation for the supported versions)
pip install torch-geometric
# install pyg-lib (check supported wheels for your CUDA version)
pip install pyg_lib 
```
### Tutorials 
SpaceTree is a versatile tool that can be used in a variety of applications. It works with both sequencing and imaging-based assays. The main difference in tutorials is the way spatial graph needs to be constructed. For technologies that are based on a grid (like Visium ST/HD) we rely on the grid for the graph construction. For technologies like Xenium, we compute the spatial similarity graph based on the spatial coordinates of the cells.

To understand the workflow please refer to our end-to-end tutorials.

For Visium/grid-based data:

- [Cell state and clone mapping to 10x Visium with SpaceTree](tutorials/cell-state-clone-mapping.md)

For Xenium data, please refer to:
- [Cell state and clone mapping to 10x Xenium with SpaceTree](tutorials/cell-state-clone-mapping-xenium.md)

We do not provide a separate tutorial for Visium HD data, as the workflow is the same as for Visium data. However, we provide some tips and tricks for working with Visium HD data [here](tutorials/visium-hd.md).

If you need help defining clones based on your own scRNA-seq data, you can use tools such as [inferCNV](https://github.com/broadinstitute/inferCNV/wiki), [inferCNVpy](https://infercnvpy.readthedocs.io/en/latest/tutorials.html), [copyKAT](https://github.com/navinlabcode/copykat) and others.

For the sake of the Visium and Xenium tutorials, we also show how we ran the clone inference based on inferCNVpy [here](https://github.com/PMBio/spaceTree/blob/master/notebooks/infercnv_run.ipynb)

