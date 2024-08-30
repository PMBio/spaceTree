<!-- markdownlint-disable -->

<a href="../spaceTree/plotting.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `plotting`





---

<a href="../spaceTree/plotting.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_xenium`

```python
plot_xenium(x, y, hue, palette=None, show=True, marker_size=1)
```

Plot Xenium. 



**Parameters:**
 
- x (array-like): The x-coordinates of the data points. 
- y (array-like): The y-coordinates of the data points. 
- hue (array-like or pd.Series): The variable used for coloring the data points. 
- palette (str or sequence, optional): The color palette to use for the plot. Defaults to None. 
- show (bool, optional): Whether to display the plot. Defaults to True. 
- marker_size (float, optional): The size of the markers. Defaults to 1. 



**Returns:**
 
- If show is True, returns None. 
- If show is False, returns a tuple containing the figure and axes objects. 


---

<a href="../spaceTree/plotting.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `confusion`

```python
confusion(xenium, label_true, label_pred)
```

Calculate and plot the confusion matrix for a given set of true and predicted labels. 



**Parameters:**
 xenium (DataFrame): The input data containing the true and predicted labels. label_true (str): The column name of the true labels. label_pred (str): The column name of the predicted labels. 



**Returns:**
 c_mat (DataFrame): The confusion matrix. avg_f1 (float): The average F1 score. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
