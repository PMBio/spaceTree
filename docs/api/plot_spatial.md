<!-- markdownlint-disable -->

<a href="../spaceTree/plot_spatial.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `plot_spatial`





---

<a href="../spaceTree/plot_spatial.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_rgb_function`

```python
get_rgb_function(cmap, min_value, max_value)
```

Generate a function to map continous values to RGB values using colormap between min_value & max_value. 


---

<a href="../spaceTree/plot_spatial.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rgb_to_ryb`

```python
rgb_to_ryb(rgb)
```

Converts colours from RGB colorspace to RYB 

Parameters 
---------- 

rgb  numpy array Nx3 

Returns 
------- Numpy array Nx3 


---

<a href="../spaceTree/plot_spatial.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `ryb_to_rgb`

```python
ryb_to_rgb(ryb)
```

Converts colours from RYB colorspace to RGB 

Parameters 
---------- 

ryb  numpy array Nx3 

Returns 
------- Numpy array Nx3 


---

<a href="../spaceTree/plot_spatial.py#L107"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_spatial_general`

```python
plot_spatial_general(
    value_df,
    coords,
    labels,
    text=None,
    circle_diameter=4.0,
    alpha_scaling=1.0,
    max_col=(inf, inf, inf, inf, inf, inf, inf),
    max_color_quantile=0.98,
    show_img=True,
    img=None,
    img_alpha=1.0,
    adjust_text=False,
    plt_axis='off',
    axis_y_flipped=True,
    axis_x_flipped=False,
    x_y_labels=('', ''),
    crop_x=None,
    crop_y=None,
    text_box_alpha=0.9,
    reorder_cmap=range(0, 7),
    style='fast',
    colorbar_position='bottom',
    colorbar_label_kw={},
    colorbar_shape={},
    colorbar_tick_size=12,
    colorbar_grid=None,
    image_cmap='Greys_r',
    white_spacing=20
)
```

Plot spatial abundance of cell types (regulatory programmes) with colour gradient and interpolation. 

 This method supports only 7 cell types with these colours (in order, which can be changed using reorder_cmap).  'yellow' 'orange' 'blue' 'green' 'purple' 'grey' 'white' 

:param value_df: pd.DataFrame - with cell abundance or other features (only 7 allowed, columns) across locations (rows) :param coords: np.ndarray - x and y coordinates (in columns) to be used for ploting spots :param text: pd.DataFrame - with x, y coordinates, text to be printed :param circle_diameter: diameter of circles :param labels: list of strings, labels of cell types :param alpha_scaling: adjust color alpha :param max_col: crops the colorscale maximum value for each column in value_df. :param max_color_quantile: crops the colorscale at x quantile of the data. :param show_img: show image? :param img: numpy array representing a tissue image.  If not provided a black background image is used. :param img_alpha: transparency of the image :param lim: x and y max limits on the plot. Minimum is always set to 0, if `lim` is None maximum  is set to image height and width. If 'no_limit' then no limit is set. :param adjust_text: move text label to prevent overlap :param plt_axis: show axes? :param axis_y_flipped: flip y axis to match coordinates of the plotted image :param reorder_cmap: reorder colors to make sure you get the right color for each category 

:param style: plot style (matplolib.style.context):  'fast' - white background & dark text;  'dark_background' - black background & white text; 

:param colorbar_position: 'bottom', 'right' or None :param colorbar_label_kw: dict that will be forwarded to ax.set_label() :param colorbar_shape: dict {'vertical_gaps': 1.5, 'horizontal_gaps': 1.5,  'width': 0.2, 'height': 0.2}, not obligatory to contain all params :param colorbar_tick_size: colorbar ticks label size :param colorbar_grid: tuple of colorbar grid (rows, columns) :param image_cmap: matplotlib colormap for grayscale image :param white_spacing: percent of colorbars to be hidden 


---

<a href="../spaceTree/plot_spatial.py#L377"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_spatial`

```python
plot_spatial(adata, color, img_key='hires', show_img=True, **kwargs)
```

Plot spatial abundance of cell types (regulatory programmes) with colour gradient and interpolation (from Visium anndata). 

This method supports only 7 cell types with these colours (in order, which can be changed using reorder_cmap). 'yellow' 'orange' 'blue' 'green' 'purple' 'grey' 'white' 

:param adata: adata object with spatial coordinates in adata.obsm['spatial'] :param color: list of adata.obs column names to be plotted :param kwargs: arguments to plot_spatial_general :return: matplotlib figure 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
