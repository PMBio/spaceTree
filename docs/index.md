<div style="text-align: center;">
  <img src="space_tree.png" alt="spacetree logo" width="200"/>
</div>

# Welcome to spaceTree's documentation!

Welcome to the official documentation for spaceTree! Here, you will find tutorials and guides to help you get started and make the most out of our tool.

### Installation
Create conda environment and install dependencies:

```bash
conda create -y -n spacetree_env python=3.9
conda activate spacetree_env
```
Finally, to use this environment in jupyter notebook, add jupyter kernel for this environment:


```bash
conda activate spacetree_env
python -m ipykernel install --user --name=spacetree_env --display-name='spacetree_env'
```

### Tutorials 
Check out our tutorials:

- [Cell state and clone mapping to 10x Visium with spaceTree](tutorials/cell-state-clone-mapping.md)

More tutorials will be added soon!
