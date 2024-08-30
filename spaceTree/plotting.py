import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.metrics import f1_score,confusion_matrix

import matplotlib as mpl
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
def plot_xenium(x, y, hue, palette=None, show=True, marker_size=1):
    """
    Plot Xenium.

    Parameters:
    - x (array-like): The x-coordinates of the data points.
    - y (array-like): The y-coordinates of the data points.
    - hue (array-like or pd.Series): The variable used for coloring the data points.
    - palette (str or sequence, optional): The color palette to use for the plot. Defaults to None.
    - show (bool, optional): Whether to display the plot. Defaults to True.
    - marker_size (float, optional): The size of the markers. Defaults to 1.

    Returns:
    - If show is True, returns None.
    - If show is False, returns a tuple containing the figure and axes objects.
    """
    x = -x
    y = -y
    if isinstance(hue, pd.Series):
        n_plots = 1
    else:
        n_plots = hue.shape[1]
    fig, axs = plt.subplots(1, n_plots, figsize=(5 * n_plots, 7))
    if n_plots != 1:
        for i in range(n_plots):
            ax = axs[i]
            if len(set(hue.iloc[:, i])) > 20:
                is_continuous = True
            else:
                is_continuous = False
            sns.scatterplot(x=y, y=x, hue=hue.iloc[:, i],
                            alpha=1, s=marker_size, palette=palette, ax=ax)
            if is_continuous:
                norm = plt.Normalize(hue.iloc[:, i].min(), hue.iloc[:, i].max())
                sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
                sm.set_array([])
                ax.get_legend().remove()
                ax.figure.colorbar(sm, ax=ax)
            else:
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            ax.axis("off")
    else:
        ax = axs
        if len(set(hue)) > 20:
            is_continuous = True
        else:
            is_continuous = False
        sns.scatterplot(x=y, y=x, hue=hue,
                        alpha=1, s=marker_size, palette=palette, ax=ax)
        if is_continuous:
            norm = plt.Normalize(hue.min(), hue.max())
            sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
            sm.set_array([])
            ax.get_legend().remove()
            ax.figure.colorbar(sm, ax=ax)
        else:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        ax.axis("off")
    ax.set_aspect('equal')
    if show:
        plt.show()
    else:
        return fig, axs
def confusion(xenium, label_true, label_pred):
    """
    Calculate and plot the confusion matrix for a given set of true and predicted labels.

    Parameters:
    xenium (DataFrame): The input data containing the true and predicted labels.
    label_true (str): The column name of the true labels.
    label_pred (str): The column name of the predicted labels.

    Returns:
    c_mat (DataFrame): The confusion matrix.
    avg_f1 (float): The average F1 score.
    """
    xenium_tmp = xenium.obs[[label_true, label_pred]].copy()
    xenium_tmp = xenium_tmp.astype("str")
    xenium_tmp.fillna("unknown", inplace=True)
    xenium_tmp = xenium_tmp.replace("nan", "unknown")
    labels = list(set(xenium_tmp[label_true]).union(set(xenium_tmp[label_pred])))
    labels.remove("unknown")
    labels.sort()
    xenium_tmp = xenium_tmp[xenium_tmp[label_true].isin(labels)]
    xenium_tmp = xenium_tmp[xenium_tmp[label_true].isin(labels)]
    c_mat = pd.DataFrame(confusion_matrix(xenium_tmp[label_true], xenium_tmp[label_pred], labels=labels, normalize='true'), index=labels, columns=labels)
    ax = sns.heatmap(c_mat, linewidth=.5, cmap="Blues", annot=True, fmt=".1f")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.show()

    def f1_from_confusion(c_mat):
        """
        Calculate the F1 score for each class based on the confusion matrix.

        Parameters:
        c_mat (DataFrame): The confusion matrix.

        Returns:
        f1 (ndarray): The F1 score for each class.
        avg_f1 (float): The average F1 score.
        """
        # Assuming c_mat is the confusion matrix
        # For a normalized confusion matrix, diagonal elements represent recall for each class
        recall = np.diag(c_mat)

        # Precision for each class can be calculated as the diagonal element divided by the sum of elements in the column
        precision = np.diag(c_mat) / np.sum(c_mat, axis=0)

        # Calculate the F1 score for each class
        f1 = 2 * precision * recall / (precision + recall)

        # If you want the average F1 score, take the mean of the F1 scores for each class
        avg_f1 = np.nanmean(f1)

        return f1, avg_f1

    f1, avg_f1 = f1_from_confusion(c_mat)
    print(avg_f1)
    return c_mat, avg_f1

