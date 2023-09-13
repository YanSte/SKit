# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2023 YanSte

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import matplotlib
import matplotlib.pyplot as plt
import math
import itertools
import numpy as np
import sklearn.metrics
import pandas as pd

from IPython.display import display,Image,Markdown,HTML

from skit.config import IS_TENSORFLOW_IMPORTED

# ==============================
#           Text
# ==============================

def show_text(heading_level, text="", add_indent=True):
    """
    Renders and displays markdown styled text, including headings of various levels or bold text.

    This function provides a convenient way to produce formatted markdown display outputs, often used within notebooks such as Jupyter.

    Parameters:
    -----------
    heading_level : str
        Specifies the style/format of the display. Supported values are:
        - 'h1' to 'h5': For headings of different levels.
        - 'b': For bold text.
        - 'sep': To display a separator line.
        Any other string will be treated as plain text.

    text : str, optional
        The content to be displayed in the specified format. Defaults to an empty string.

    add_indent : bool, optional
        If set to True, an additional line break will be added after the displayed text, providing an indentation effect. Defaults to True.

    Example:
    --------
    >>> show_text('h1', 'Title')
    # Title (displayed as a Level 1 Heading)

    >>> show_text('b', 'Bolded Text')
    **Bolded Text** (displayed as bold text)

    >>> show_text('sep')
    ==========================================
    """
    if heading_level == 'h1':
        display(Markdown(f'# {text}'))

    elif heading_level == 'h2':
        display(Markdown(f'## {text}'))

    elif heading_level == 'h3':
        display(Markdown(f'### {text}'))

    elif heading_level == 'h4':
        display(Markdown(f'#### {text}'))

    elif heading_level == 'h5':
        display(Markdown(f'##### {text}'))

    elif heading_level == 'b':
        display(Markdown(f'**{text}**'))

    elif heading_level == 'sep':
        print("=" * 50)

    else:
        display(Markdown(f'{text}'))

    if add_indent:
        print("\n")

# ==============================
#           History
# ==============================

def show_best_history(
    history,
    metric="val_acc",
    add_metric=["acc"]
):
    """
    Summarize the best model performance based on training history.

    Args:
        history :dict
            The training history dictionary returned by Keras.
        primary_metric : str
            The primary metric to monitor for the best value (default: "val_acc").
        additional_metrics : list
            List of additional metrics to extract complementary values (default: ["acc"]).
    """
    history = history.history

    if metric not in history:
        print(f"The metric '{metric}' not in history.")
        return

    for metric in add_metric:
        if metric not in history:
            print(f"The metric '{metric}' not in history.")
            return

    best_index = np.argmax(history[metric])
    best_value = history[metric][best_index]

    result = f"Best {metric}: {best_value:.2f}  with "

    for i, metric in enumerate(add_metric):
        best_value = history[metric][best_index]
        result += f"{metric}: {best_value:.2f} "
        if i < len(add_metric) - 1:
            result += ", "

    show_text("b", result)

def show_history(
    history,
    title = "Accurancy",
    y_label = "Accurancy",
    metrics = ["acc", "val_acc"],
    metric_labels = ["Train Accurancy", "Validation Accurancy"],
    figsize = (8,6)
):
    """
    Visualizes the training and validation metrics from the model's history using matplotlib.

    The function generates separate plots for each metric defined in the `metrics` parameter.
    It uses the corresponding labels provided in the `metric_labels` parameter for the plot legends.

    Parameters:
    -----------
    history : dict
        The history object typically returned from the .fit() method of a Keras model. It should
        have a 'history' attribute containing the training and validation metrics.

    title : str, optional
        The title for the plot. Defaults to "Accuracy".

    metrics : list of str, optional
        A list of metric names to be plotted. Defaults to ["acc", "val_acc"].

    metric_labels : list of str, optional
        A list of labels for the plotted metrics. Should have the same length as `metrics`.
        Defaults to ["Accuracy", "Validation Accuracy"].

    figsize : tuple, optional
        The width and height in inches for the figure. Defaults to (8, 6).

    Example:
    --------
    show_history(
        model_history,
        title="Training and Validation Loss",
        metrics=["loss", "val_loss"],
        metric_labels=["Training Loss", "Validation Loss"],
        figsize=(10, 8)
    )
    """
    history = history.history

    for metric in metrics:
        if metric not in history:
            print(f"The metric '{metric}' not in history.")
            return

    plt.figure(figsize=figsize)
    plt.title(title)

    plt.ylabel(y_label)
    plt.xlabel('Epoch')

    for metric_name, metric_label in zip(metrics, metric_labels):
        plt.plot(history[metric_name], label=metric_label)

    plt.legend(loc='upper left')
    plt.show()

# ==============================
#           Image
# ==============================

def show_images(
    x,
    y=None,
    indices='all',
    columns=12,
    figure_size=(1, 1),
    show_colorbar=False,
    y_pred=None,
    color_map='binary',
    normalization=None,
    padding=0.35,
    spines_alpha=1,
    font_size=20,
    interpolation='lanczos'
):
    """
    Displays a grid of images with their corresponding labels.

    This function provides a visually pleasing way to visualize images, typically used for exploratory data analysis in notebook environments.

    Parameters:
    -----------
    x : numpy.ndarray
        The array of images to display. Supported shapes are (-1, lx, ly), (-1, lx, ly, 1), or (-1, lx, ly, 3).

    y : list or numpy.ndarray, optional
        The true class labels or annotations associated with the images.

    indices : list or str, optional
        Specifies which images to display from the input 'x'. If set to 'all', all images will be displayed. Otherwise, it should be a list of indices. Defaults to 'all'.

    columns : int, optional
        Number of columns in the grid layout of images. Defaults to 12.

    figure_size : tuple of int, optional
        Specifies the width and height of each individual figure in the grid. Defaults to (1, 1).

    show_colorbar : bool, optional
        Whether to display the colorbar beside the images. This can be helpful for grayscale images. Defaults to False.

    y_pred : list or numpy.ndarray, optional
        Predicted class labels for the images.

    color_map : str, optional
        Name of the colormap used by matplotlib to plot the images. Defaults to 'binary'.

    normalization : tuple of float or None, optional
        Tuple specifying the normalization bounds as (min, max) for displaying the image. Useful for enhancing contrast in certain images. Defaults to None.

    padding : float, optional
        Specifies the padding between rows in the grid. Defaults to 0.35.

    spines_alpha : float, optional
        Transparency level of the borders around the images. Ranges between 0 (completely transparent) to 1 (fully opaque). Defaults to 1.

    font_size : int, optional
        Font size for the labels. Defaults to 20.

    interpolation : str, optional
        Specifies the interpolation algorithm to use when displaying the images. Useful for high-resolution or upscaled images. Defaults to 'lanczos'.

    Example:
    --------
    >>> images = np.random.random((100, 28, 28))
    >>> labels = ["Label" + str(i) for i in range(100)]
    >>> show_images(images, y=labels, columns=10, figure_size=(2, 2))

    Notes:
    ------
    If both 'y' (true labels) and 'y_pred' (predicted labels) are provided, and they differ for an image, the predicted label will be shown in red followed by the true label in parentheses.
    """
    if indices == 'all':
        indices = range(len(x))

    if normalization and len(normalization) == 2:
        normalization = matplotlib.colors.Normalize(vmin=normalization[0], vmax=normalization[1])

    draw_labels = (y is not None)
    draw_predicted_labels = (y_pred is not None)

    rows = math.ceil(len(indices) / columns)
    fig = plt.figure(figsize=(columns * figure_size[0], rows * (figure_size[1] + padding)))

    n = 1
    for i in indices:
        axs = fig.add_subplot(rows, columns, n)
        n += 1

        # Shape is (lx,ly)
        # ----
        if len(x[i].shape) == 2:
            xx = x[i]
        # Shape is (lx,ly,n)
        # ----
        if len(x[i].shape) == 3:
            (lx, ly, lz) = x[i].shape
            if lz == 1:
                xx = x[i].reshape(lx, ly)
            else:
                xx = x[i]

        img = axs.imshow(xx, cmap=color_map, norm=normalization, interpolation=interpolation)

        axs.spines['right'].set_visible(True)
        axs.spines['left'].set_visible(True)
        axs.spines['top'].set_visible(True)
        axs.spines['bottom'].set_visible(True)

        axs.spines['right'].set_alpha(spines_alpha)
        axs.spines['left'].set_alpha(spines_alpha)
        axs.spines['top'].set_alpha(spines_alpha)
        axs.spines['bottom'].set_alpha(spines_alpha)

        axs.set_yticks([])
        axs.set_xticks([])

        if draw_labels and not draw_predicted_labels:
            axs.set_xlabel(y[i], fontsize=font_size)

        if draw_labels and draw_predicted_labels:
            if y[i] != y_pred[i]:
                axs.set_xlabel(f'{y_pred[i]} (✓: {y[i]})', fontsize=font_size)
                axs.xaxis.label.set_color('red')
            else:
                axs.set_xlabel(y[i], fontsize=font_size)

        if show_colorbar:
            fig.colorbar(img, orientation="vertical", shrink=0.65)

    plt.show()

# ==============================
#           Donut
# ==============================

def show_donut(
    values,
    labels,
    colors=["lightsteelblue","coral"],
    figsize=(6,6),
    title=None
):
    """
    Displays a donut chart.

    Parameters:
    -----------
    values : list
        List of values corresponding to each segment in the donut chart.

    labels : list
        List of labels for each segment.

    colors : list, optional
        List of colors for each segment. Default is ["lightsteelblue", "coral"].

    figsize : tuple, optional
        Size of the figure. Default is (6, 6).

    title : str, optional
        Title for the donut chart. Default is None.
    """
    # ---- Donut
    plt.figure(figsize=figsize)
    # ---- Draw a pie  chart..
    plt.pie(values, labels=labels,
            colors = colors, autopct='%1.1f%%', startangle=70, pctdistance=0.85,
            textprops={'fontsize': 18},
            wedgeprops={"edgecolor":"w",'linewidth': 5, 'linestyle': 'solid', 'antialiased': True})
    # ---- ..with a white circle
    circle = plt.Circle((0,0),0.70,fc='white')
    ax = plt.gca()
    ax.add_artist(circle)
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show()

# ==============================
#      Confusion Matrix
# ==============================

def show_confusion_matrix(
    y_true,
    y_pred,
    labels,
    title='Confusion matrix',
    cmap=None,
    normalize=True,
    figsize=(10, 8),
    digit_format='{:0.2f}'
):
    """
    Displays a confusion matrix.

    Parameters:
    -----------
    y_true : array-like
        List or array of true labels.

    y_pred : array-like
        List or array of predicted labels.

    labels : list
        List of labels used in confusion matrix.

    title : str, optional
        Title for the confusion matrix. Default is 'Confusion matrix'.

    cmap : colormap, optional
        Color map used for plotting. Default is None, which means 'Blues' colormap will be used.

    normalize : bool, optional
        If True, the confusion matrix will be normalized. Default is True.

    figsize : tuple, optional
        Size of the figure. Default is (10, 8).

    digit_format : str, optional
        Format for the numbers in the confusion matrix. Default is '{:0.2f}'.
    """
    cm = sklearn.metrics.confusion_matrix( y_true,y_pred, normalize=None, labels=labels)

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if labels is not None:
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=90)
        plt.yticks(tick_marks, labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, digit_format.format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

# ==============================
#          Histogram
# ==============================

def show_histogram(
    data,
    bins=10,
    range=None,
    xlabel=None,
    ylabel=None,
    title=None,
    alpha=1.0,
    color='lightsteelblue',
    edgecolor='black'
):
    """
    Plot a histogram.

    Args:
        data : list or Series
            The data to create the histogram from.
        bins : int or sequence, optional
            The number of bins or the specific bin edges.
        range : tuple, optional
            The range of values to consider for the histogram.
        xlabel : str, optional
            Label for the x-axis.
        ylabel :str, optional
            Label for the y-axis.
        title : str, optional
            Title for the histogram.
        alpha : float, optional
            Transparency of the bars (0.0 to 1.0).
        color : str, optional
            Color of the bars.
    """
    plt.hist(data, bins=bins, range=range, alpha=alpha, color=color, edgecolor=edgecolor)

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)

    plt.show()


if IS_TENSORFLOW_IMPORTED:
    import tensorflow as tf

    def tf_show_images(
        dataset,
        labels="default", # default get from dataset
        indices="all",
        columns=1,
        figure_size=(1, 1),
        show_colorbar=False,
        pred=None,
        color_map='binary',
        normalization=None,
        padding=0.35,
        spines_alpha=1,
        font_size=20,
        interpolation='lanczos'
    ):
        """
        Displays images from a TensorFlow dataset.

        Parameters:
        -----------
        dataset : tf.data.Dataset
            TensorFlow dataset containing images and labels.

        labels : list or str, optional
            List of labels for images. If set to 'default', the labels are fetched from the dataset.
            Default is 'default'.

        indices : list or str, optional
            Indices of the images to display from the dataset. If set to 'all', all images are displayed.
            Default is 'all'.

        columns : int, optional
            Number of columns for displaying images. Default is 1.

        figure_size : tuple, optional
            Size of the figure for each image. Default is (1, 1).

        show_colorbar : bool, optional
            If True, a color bar is shown alongside the images. Default is False.

        pred : array-like, optional
            List or array of predicted labels for images.

        color_map : str, optional
            Color map used for plotting images. Default is 'binary'.

        normalization : str, optional
            Type of normalization to apply to images. Currently unused and kept for future extension.
            Default is None.

        padding : float, optional
            Padding around images. Default is 0.35.

        spines_alpha : float, optional
            Alpha value for spines around images. Default is 1.

        font_size : int, optional
            Font size for labels. Default is 20.

        interpolation : str, optional
            Interpolation method used for displaying images. Default is 'lanczos'.

        Raises:
        -------
        Exception:
            - If the dataset does not have a 'class_names' attribute and labels are set to 'default'.
            - If the specified number of columns is greater than the size of the dataset.
            - If the specified indices are greater than the batch size.

        Notes:
        ------
        This function is designed to work with TensorFlow datasets. Ensure you have TensorFlow imported and
        the dataset provided is compatible.
        """
        if not isinstance(dataset, tf.data.Dataset):
            raise ValueError("The provided dataset is not an instance of tf.data.Dataset.")

        # Labels
        # ----
        if labels == "default":
            try:
                class_names = dataset.class_names
            except AttributeError:
                raise Exception(
                    f"""
                    The provided dataset of type {type(dataset)} does not have a 'class_names' attribute.
                    This usually occurs when the dataset is not a 'BatchDataset' or a dataset preprocessed by certain high-level APIs.
                    If you are using a custom dataset type, please specify the 'labels' argument.
                    """
                )

        # Columns size
        # ----
        dataset_size = dataset.cardinality().numpy()

        if columns <= dataset_size:
            dataset = dataset.take(columns)
        else:
            raise Exception(f"The columns is bigger than the dataset size: {dataset_size}.")

        # Setup x y
        # ----
        x = []
        y = []

        for images, labels_mapping in dataset:
            x.extend(images.numpy())
            y.extend(np.argmax(labels_mapping.numpy(), axis=1))  # get class numbers

        x = np.array(x)
        y = [labels[i] for i in y]  # convert to class names

        # Row size
        # ----
        batch_size = len(x)
        if indices == "all":
            indices = range(0, batch_size)
        else:
            if indices <= len(x):
                indices = range(0, indices)
            else:
                raise Exception(f"The indices is bigger than batch size: {batch_size}.")

        # Show
        # ----
        show_images(
            x,
            y,
            figure_size=(3, 3),
            indices=indices,
            show_colorbar=show_colorbar,
            y_pred=pred,
            color_map=color_map,
            padding=padding,
            spines_alpha=spines_alpha,
            font_size=font_size,
            interpolation=interpolation
        )
