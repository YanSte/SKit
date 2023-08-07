import numpy as np

from skit.config import IS_TENSORFLOW_IMPORTED

# ==============================
#           TensorFlow
# ==============================

if IS_TENSORFLOW_IMPORTED:
    import tensorflow as tf

    def tf_predictions(dataset, model, num_take='all'):
        """
        Generates predictions on a provided TensorFlow dataset using the given model.

        Parameters:
        -----------
        dataset : tf.data.Dataset
            The input dataset for which predictions are required. This should contain data samples and their corresponding labels.

        model : tf.keras.Model
            The TensorFlow model to be used for generating predictions.

        num_take : int or 'all', optional
            Number of samples from the dataset on which predictions are to be made. If set to 'all', predictions will be generated for the entire dataset. Defaults to 'all'.

        Returns:
        --------
        tuple of numpy.ndarray
            Three arrays are returned:
            - The first array contains the input data samples.
            - The second array contains the true labels.
            - The third array contains the predicted values from the model.

        Raises:
        -------
        ValueError:
            If the input 'dataset' is not an instance of tf.data.Dataset.

        Exception:
            If the value of 'num_take' is greater than the number of samples available in the dataset.

        Notes:
        ------
        This function assumes that the input dataset is already batched. If the dataset contains very large images or is not batched, the function might consume a large amount of memory.

        Example:
        --------
        >>> dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
        >>> model = tf.keras.models.load_model('path_to_model.h5')
        >>> x, y_true, y_pred = tf_predictions(dataset, model)
        """
        if not isinstance(dataset, tf.data.Dataset):
            raise ValueError("The provided dataset is not an instance of tf.data.Dataset.")

        # Num take
        # ----
        if num_take != 'all':
            dataset_size = dataset.cardinality().numpy()
            if num_take > dataset_size:
                raise Exception(f"The num_take is bigger than the dataset size: {dataset_size}.")
            else:
                dataset = dataset.take(num_take)

        # Extraire x_test et y_test du tf.data.Dataset
        x_test = []
        y_test = []
        y_pred = []

        for data, label in dataset.as_numpy_iterator():
            x_test.append(data)
            y_test.append(label)

            predictions = model.predict_on_batch(data).flatten()
            y_pred.append(predictions)

        x_test = np.concatenate(x_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        return x_test, y_test, y_pred

    def tf_convert_predictions_to_labels(y_test, y_pred, class_labels, loss_type="categorical_crossentropy"):
        """
        Converts y_test and predicted probabilities in y_pred_values to their corresponding class labels.

        Parameters:
        -----------
        y_test : numpy.ndarray
            Array of true class labels. One-hot encoded for 'categorical_crossentropy' and integer-encoded for 'sparse_categorical_crossentropy'.

        y_pred : numpy.ndarray
            Array of predicted class probabilities. This can be obtained from a model's prediction output.

        class_labels : list of str
            List of class labels where the index of each label corresponds to its class index.

        loss_type : str, optional
            Type of loss used in the model. Either 'categorical_crossentropy' or 'sparse_categorical_crossentropy'. Defaults to 'categorical_crossentropy'.

        Returns:
        --------
        tuple of list of str
            Two lists are returned:
            - The first list contains the true class labels.
            - The second list contains the predicted class labels based on the max probability.

        Example:
        --------
        >>> y_true_labels, y_pred_labels = tf_convert_predictions_to_labels(y_test, y_pred, ["cat", "dog", "bird"], loss_type="sparse_categorical_crossentropy")
        """
        if loss_type == "categorical_crossentropy":
            y_test_indices = [np.argmax(row) for row in y_test]
        elif loss_type == "sparse_categorical_crossentropy":
            y_test_indices = y_test.astype(int)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        y_test_labels = [class_labels[idx] for idx in y_test_indices]

        y_pred_indices = [np.argmax(row) for row in y_pred]
        y_pred_labels = [class_labels[idx] for idx in y_pred_indices]

        return y_test_labels, y_pred_labels
