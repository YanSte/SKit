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
#           Show
# ==============================

def show_text(heading_level, text="", add_indent=True):
    '''
    Display a markdown heading or bold text
    args:
        heading_level : heading level (h1, h2, h3, h4, h5, b)
        text : text to display
    return:
        none
    '''
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

def show_history(
    history,
    figsize = (8,6),
    plot = {
        'Accuracy': {
            'Training Accuracy': 'accuracy',
            'Validation Accuracy': 'val_accuracy'
        },
        'Loss': {
            'Training Loss': 'loss',
            'Validation Loss': 'val_loss'
        }
    }
):
    """
    Visualizes the training and validation metrics from the model's history using matplotlib.

    The function generates separate plots for each main category (like 'Accuracy' and 'Loss')
    defined in the `plot` parameter. For each main category, multiple curves (like 'Training Accuracy'
    and 'Validation Accuracy') can be plotted based on the nested dictionary values.

    Parameters:
    -----------
    history : dict
        The history object typically returned from the .fit() method of a Keras model. It should
        have a 'history' attribute containing the training and validation metrics.

    figsize : tuple, optional
        The width and height in inches for the figure. Defaults to (8,6).

    plot : dict, optional
        A nested dictionary defining the metrics to be plotted.
        - The top-level key corresponds to the main category (e.g., 'Accuracy' or 'Loss').
        - The associated nested dictionary's keys are the curve labels (e.g., 'Training Accuracy')
          and the values are the corresponding metric names in the 'history' object (e.g., 'accuracy').
        Defaults to plotting both training and validation accuracy and loss.

    Example:
    --------
    show_history(
        model_history,
        figsize=(10,8),
        plot={
            "Titre A": {
                "Legend Titre 1": "metric",
                "Legend Titre 2": "metric"
                }
            }
    )
    """
    for title, curves in plot.items():
        plt.figure(figsize=figsize)
        plt.title(title)

        # Extracting the name from the first metric and capitalizing the first letter for ylabel
        y_label = list(curves.values())[0].capitalize()
        plt.ylabel(y_label)
        plt.xlabel('Epoch')

        for curve_label, metric_name in curves.items():
            plt.plot(history.history[metric_name], label=curve_label)
        plt.legend(loc='upper left')
        plt.show()

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
    Display a grid of images with labels.

    Args:
        x: The images to display. Shapes must be (-1, lx, ly), (-1, lx, ly, 1), or (-1, lx, ly, 3).
        y: Real classes or labels associated with the images. (None)
        indices: Indices of images to show or 'all' for all images. ('all')
        columns: Number of columns in the grid. (12)
        figure_size: Size of the figure (width, height). (1, 1)
        show_colorbar: Whether to show the colorbar. (False)
        y_pred: Predicted classes associated with the images. (None)
        color_map: Matplotlib color map to use. ('binary')
        normalization: Matplotlib imshow normalization. (None)
        padding: Padding between rows in the grid. (0.35)
        spines_alpha: Alpha value for the spines. (1)
        font_size: Font size in pixels. (20)
        interpolation: Interpolation method for displaying the images. ('lanczos')

    Returns:
        None
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
                axs.set_xlabel(f'{y_pred[i]} ({y[i]})', fontsize=font_size)
                axs.xaxis.label.set_color('red')
            else:
                axs.set_xlabel(y[i], fontsize=font_size)

        if show_colorbar:
            fig.colorbar(img, orientation="vertical", shrink=0.65)

    plt.show()


def show_donut(
    values,
    labels,
    colors=["lightsteelblue","coral"],
    figsize=(6,6),
    title=None
):
    """
    Draw a donut
    args:
        values   : list of values
        labels   : list of labels
        colors   : list of color (["lightsteelblue","coral"])
        figsize  : size of figure ( (6,6) )
    return:
        nothing
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

def show_confusion_matrix(
    y_true,
    y_pred,
    target_names,
    title='Confusion matrix',
    cmap=None,
    normalize=True,
    figsize=(10, 8),
    digit_format='{:0.2f}'
):
    cm = sklearn.metrics.confusion_matrix( y_true,y_pred, normalize=None, labels=target_names)

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

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
#           TensorFlow
# ==============================

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
