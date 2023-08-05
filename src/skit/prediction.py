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
        if not isinstance(dataset, tf.data.Dataset) == False:
          raise Exception("The dataset is not a 'tf.data.Dataset' instance.")

        x_test = []
        y_test = []
        y_pred = []

        # Labels
        # ----
        if labels == "default":
            labels = dataset.class_names

        # Num take
        # ----
        if num_take != 'all':
          try:
            dataset = dataset.take(num_take)
          else:
            dataset_size = dataset.cardinality().numpy()
            raise Exception(f"The num_take is bigger than the dataset size: {dataset_size}.")

        for images, true_label in dataset:
            # Get feature
            # ----
            x_test.extend(images.numpy())

            # Get true label
            # ----
            true_label_scalar = true_label.numpy()
            true_index = np.where(true_label_scalar == 1)[0][0]
            y_test.append(labels[true_index])

            predictions = model.predict(images, verbose=verbosity)

            for pred in predictions:
                # Get predicted label
                # ----
                predicted_score = tf.nn.softmax(pred)
                predicted_label = labels[np.argmax(predicted_score)]
                y_pred.append(predicted_label)

        return x_test, y_test, y_pred
