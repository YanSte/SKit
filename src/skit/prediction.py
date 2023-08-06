import numpy as np

from skit.config import IS_TENSORFLOW_IMPORTED

# ==============================
#           TensorFlow
# ==============================

if IS_TENSORFLOW_IMPORTED:
    import tensorflow as tf

    def tf_predictions(dataset, model, num_take='all', labels='default', verbosity=0):
        """
        Makes predictions using a TensorFlow model on a given dataset and returns the test data,
        test labels and predicted labels.

        Parameters:
        - dataset (tf.data.Dataset): The dataset to make predictions on. Must be an instance of tf.data.Dataset.
        - model (tf.keras.Model): The model to use for making predictions.
        - num_take (int or str, optional): The number of data points to take from the dataset for making predictions. If 'all', all data points will be used. Defaults to 'all'.
        - labels (list or str, optional): The list of class names. If 'default', will try to use dataset.class_names. Defaults to 'default'.
        - verbosity (int, optional): Verbosity mode, 0 or 1. Defaults to 0.

        Raises:
        - Exception: If the dataset is not a tf.data.Dataset instance.
        - Exception: If num_take is not 'all' and is larger than the size of the dataset.
        - Exception: If labels is 'default' but class names cannot be obtained from the dataset.

        Returns:
        - tuple: (x_test, y_test, y_pred) where
            - x_test (list): The test data used for making predictions.
            - y_test (list): The actual labels of the test data.
            - y_pred (list): The predicted labels.
        """
        if not isinstance(dataset, tf.data.Dataset):
            raise ValueError("The provided dataset is not an instance of tf.data.Dataset.")

        # Labels
        # ----
        if labels == "default":
            try:
                labels = dataset.class_names
            except AttributeError:
                raise AttributeError("The dataset does not have an attribute 'class_names'. Please provide explicit labels.")

        # Num take
        # ----
        if num_take != 'all':
            dataset_size = dataset.cardinality().numpy()
            if num_take > dataset_size:
                raise ValueError(f"The value of num_take ({num_take}) exceeds the dataset size: {dataset_size}.")
            dataset = dataset.take(num_take)

        # Setup returns
        # ----
        x_test = []
        y_test = []
        y_pred = []

        for images, true_labels in dataset:
            # Get feature
            # ----
            x_test.extend(images.numpy())

            # Get true label
            # ----
            y_test.extend(labels[idx] for idx in np.argmax(true_labels.numpy(), axis=1))

            # Predict
            # ----
            predictions = model.predict(images, verbose=verbosity)

            # Get predicted labels
            # ----
            y_pred.extend(labels[idx] for idx in np.argmax(predictions, axis=1))

        return x_test, y_test, y_pred
