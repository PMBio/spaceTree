---
layout: default
title: Tutorials
nav_order: 2
has_children: true
---
# Tutorials

Welcome to the tutorials section. Here, you can find various guides to help you with different aspects of spaceTree.

spaceTree is a versatile tool that can be used in a variety of applications. It works with both sequencing and imaging-based assays. The main difference in tutorials is the way spatial graph needs to be constructed. For technologies that are based on a grid (like Visium ST/HD) we rely on the grid for the graph construction. For technologies like Xenium, we compute the spatial similarity graph based on the spatial coordinates of the cells.

To understand the workflow please refer to our end-to-end tutorials.

For Visium/grid-based data:

- [Cell state and clone mapping to 10x Visium with spaceTree](cell-state-clone-mapping.md)

For Xenium data, please refer to:
- [Cell state and clone mapping to 10x Xenium with spaceTree](cell-state-clone-mapping-xenium.md)

We do not provide a separate tutorial for Visium HD data, as the workflow is the same as for Visium data. However, we provide some tips and tricks for working with Visium HD data [here](visium-hd.md).

If you need help defining clones based on your own scRNA-seq data, you can use tools such as [inferCNV](https://github.com/broadinstitute/inferCNV/wiki), [inferCNVpy](https://infercnvpy.readthedocs.io/en/latest/tutorials.html), [copyKAT](https://github.com/navinlabcode/copykat) and others.

For the sake of the Visium and Xenium tutorials, we also show how we ran the clone inference based on inferCNVpy [here](https://github.com/PMBio/spaceTree/blob/master/notebooks/infercnv_run.ipynb)