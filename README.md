<div style="text-align: center;">
  <img src="docs/space_tree.png" alt="spacetree logo" width="200"/>
</div>

SpaceTree: Deciphering Tumor Microenvironments by joint modeling of cell states and genotype-phenotype relationships in spatial omics data 
==============================

SpaceTree jointly models spatially smooth cell type- and clonal state composition.
SpaceTree employs Graph Attention mechanisms, capturing information from spatially close regions when reference mapping falls short, enhancing both interpretation and quantitative accuracy. 

A significant merit of SpaceTree is its technology-agnostic nature, allowing clone-mapping in sequencing- and imaging-based assays. 
The model outputs can be used to characterize spatial niches that have consistent cell type and clone composition.


<div style="text-align: center;">
  <img src="docs/schema.jpg" alt="spacetree logo" width="1000"/>
</div>
Overview of the spatial mapping approach and the workflow enabled by SpaceTree.From left to right: SpaceTree requirs as input reference (scRNA-seq) and spatial count matrices as well as labels that need to be transfered. The labels can be descrete, continious or hierachical. The model outputs a spatial mapping of the labels and the cell type (compositions in case of Visium) of the spatial regions.

## Usage and Tutorials


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
If you struggle with compiling the `pyg-lib` library, you can also use precompiled wheels from [here](https://data.pyg.org/whl/) and then install with `pip install {file_name}.whl`
### Documentation, Tutorials and Examples
Check out our tutorials and documentation to get started with spaceTree [here](https://pmbio.github.io/spaceTree/).

## Citation
