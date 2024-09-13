---
layout: default
title: FAQ
nav_order: 4
has_children: false
---

# FAQ
- **Q:** What if I want to map only cell states, but not clones?
- **A:** In therory, spaceTree can be used with any type of labels that can be transfered from scRNA-seq to spatial data. In practice, a way to achive that at the moment requirs a few modifications to the tutorial:
-----------------

1) Currently the format of `data` object is fixed, but you can assign a dummy label to `clone` column in `adata.obs` by assigning roughlhly 50% of the cells to clone '0' and the rest to 'diploid'.

2) When you define the model, you need to set `map_enteties` to 'type' instead of 'both'. E.g.:
```python
model = GATLightningModule_sampler(data_param=data, weight_type=weight_type, map_enteties='type')
```
3) Instead of running `get_results_calibrated()`, you can use `get_results_type(pred, data, node_encoder_rev, node_encoder_ct, activation='softmax')` to get the results for cell types only. 
-----------------
- **Q:** What if I want to map only clones, but not cell types?

- **A:** In therory, spaceTree can be used with any type of labels that can be transfered from scRNA-seq to spatial data. In practice, a way to achive that at the moment requirs a few modifications to the tutorial:
-----------------

1) Currently the format of `data` object is fixed, but you can assign a dummy label to 'cell_type' column in `adata_ref.obs` by assigning roughlhly 50% of the cells to dummy_cell_type1 and the rest to 'dummy_cell_type2' (or use any other names).

2) When you define the model, you need to set `map_enteties` to 'clone' instead of 'both'. E.g.:
```python
model = GATLightningModule_sampler(data_param=data, weight_clone=weight_clone, map_enteties='clone')
```
3) Instead of running `get_results_calibrated()`, you can use `get_results_clone(pred, data, node_encoder_rev, node_encoder_cl, activation='softmax')` to get the results for clones only. 
-----------------

- **Q:** What if I want to map a different type of labels, e.g. cell types and cell cycle phases?
- **A:** In therory, spaceTree can be used with any type of labels that can be transfered from scRNA-seq to spatial data. In practice, due to due to the fixed format of clonal labels, you might have to represent cell cycle phases as clones: e.g. "diploid","0","1", etc.
-----------------

- **Q:** What if I want to use more than 2 sets of labels?
- **A:** Right now we do not support more than 2 sets of labels. However, if you feel confident in your programming skills, you can modify the code of model and the evaluation functions to support more than 2 sets of labels.

-----------------

