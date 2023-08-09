import numpy as np
from tqdm import tqdm
from skit.config import IS_TENSORFLOW_IMPORTED

# ==============================
#           TensorFlow
# ==============================

if IS_TENSORFLOW_IMPORTED:
    import tensorflow as tf

    def tf_predictions(dataset, model, labels, with_x_test=False, loss_type="categorical_crossentropy", verbosity=0):
        """
        Generates predictions on a provided TensorFlow dataset using the given model.

        Parameters:
        -----------
        dataset : tf.data.Dataset
            The input dataset for which predictions are required. This should contain data samples and their corresponding labels.

        model : tf.keras.Model
            The TensorFlow model to be used for generating predictions.

        labels : list of str
            List of class labels where the index of each label corresponds to its class index.

        with_x_test : bool, optional
            If set to True, the returned tuple will contain the input data samples as the first element. Defaults to False.

        loss_type : str, optional
            Type of loss used in the model. Either 'categorical_crossentropy' or 'sparse_categorical_crossentropy'. Defaults to 'categorical_crossentropy'.

        verbosity : int, optional
            Verbosity mode. 0 = silent, 1 = progress bar. Defaults to 0.

        Returns:
        --------
        tuple of numpy.ndarray
            If with_x_test is True:
            - The first array contains the input data samples.
            - The second array contains the true labels.
            - The third array contains the predicted values from the model.

            If with_x_test is False:
            - The first array contains the true labels.
            - The second array contains the predicted values from the model.

        Raises:
        -------
        ValueError:
            If the input 'dataset' is not an instance of tf.data.Dataset.

        MemoryError:
            If memory is exceeded while processing data.

        Notes:
        ------
        This function assumes that the input dataset is already batched. If the dataset contains very large images or is not batched, the function might consume a large amount of memory.

        Example:
        --------
        >>> dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
        >>> model = tf.keras.models.load_model('path_to_model.h5')
        >>> x, y_true, y_pred = tf_predictions(dataset, model, labels=["cat", "dog", "bird"], with_x_test=True)
        """
        if not isinstance(dataset, tf.data.Dataset):
            raise ValueError("The provided dataset is not an instance of tf.data.Dataset.")

        # Extraire x_test et y_test du tf.data.Dataset
        if with_x_test:
            x_test = []

        y_test = []
        y_pred = []

        try:
            for data, label in tqdm(dataset, desc="Predicting", unit="batch"):
                if with_x_test:
                    x_test.append(data)

                y_test.append(label)

                predictions = model.predict(data, verbose=verbosity)
                y_pred.append(predictions)

            if with_x_test:
                x_test = np.concatenate(x_test, axis=0)

            y_test = np.concatenate(y_test, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)

            y_test, y_pred = tf_convert_predictions_to_labels(y_test, y_pred, labels)

        except MemoryError:
            raise MemoryError("Memory exceeded while processing data. Consider processing a smaller dataset with take.")

        if with_x_test:
            return x_test, y_test, y_pred
        else:
            return y_test, y_pred

    def tf_wrong_predictions(dataset, model, labels, qt_desired, with_x_test=False, loss_type="categorical_crossentropy", verbosity=0):
        """
        Generates a subset of predictions on a provided TensorFlow dataset using the given model,
        but only collects those where the model's predictions were incorrect.

        Parameters:
        -----------
        dataset : tf.data.Dataset
            The input dataset for which predictions are required. This should contain data samples and their corresponding labels.

        model : tf.keras.Model
            The TensorFlow model to be used for generating predictions.

        labels : list of str
            List of class labels where the index of each label corresponds to its class index.

        qt_desired : int
            The number of incorrect predictions desired. The function will stop collecting predictions once this number is reached.

        with_x_test : bool, optional
            If set to True, the returned tuple will contain the input data samples as the first element. Defaults to False.

        loss_type : str, optional
            Type of loss used in the model. Either 'categorical_crossentropy' or 'sparse_categorical_crossentropy'. Defaults to 'categorical_crossentropy'.

        verbosity : int, optional
            Verbosity mode. 0 = silent, 1 = progress bar. Defaults to 0.

        Returns:
        --------
        tuple of numpy.ndarray
            If with_x_test is True:
            - The first array contains the input data samples.
            - The second array contains the true labels.
            - The third array contains the predicted values from the model.

            If with_x_test is False:
            - The first array contains the true labels.
            - The second array contains the predicted values from the model.

        Raises:
        -------
        ValueError:
            If the input 'dataset' is not an instance of tf.data.Dataset.

        Notes:
        ------
        This function assumes that the input dataset is already batched. If the dataset contains very large images or is not batched, the function might consume a large amount of memory.
        It specifically focuses on gathering incorrect predictions, based on the qt_desired parameter.

        Example:
        --------
        >>> dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
        >>> model = tf.keras.models.load_model('path_to_model.h5')
        >>> x, y_true, y_pred = tf_wrong_predictions(dataset, model, labels=["cat", "dog", "bird"], qt_desired=10, with_x_test=True)
        """
        if not isinstance(dataset, tf.data.Dataset):
            raise ValueError("The provided dataset is not an instance of tf.data.Dataset.")

        if with_x_test:
            x_test = []

        y_test = []
        y_pred = []

        incorrect_count = 0

        # Initialize the tqdm progress bar
        pbar = tqdm(total=qt_desired, desc="Collecting incorrect predictions", unit="sample")

        for data, label in dataset:
            predictions = model.predict(data, verbose=verbosity)

            # Convert the predictions and labels to class labels
            true_labels, pred_labels = tf_convert_predictions_to_labels(label.numpy(), predictions, labels, loss_type)

            for i, (true_label, pred_label) in enumerate(zip(true_labels, pred_labels)):
                # Only append if the prediction is incorrect
                if true_label != pred_label:
                    if with_x_test:
                        x_test.append(data.numpy()[i])  # Append individual samples to the list

                    y_test.append(true_label)
                    y_pred.append(pred_label)

                    # Increment the incorrect count
                    incorrect_count += 1

                    # Update the tqdm progress bar
                    pbar.update(1)  # Increment by one sample

                    # Check if we've reached the desired number of incorrect predictions
                    if incorrect_count >= qt_desired:
                        break

            # Check again outside the inner loop
            if incorrect_count >= qt_desired:
                break

        # Show and close the progress bar when done
        pbar.update(qt_desired)
        pbar.close()

        if with_x_test:
            return np.array(x_test), np.array(y_test), np.array(y_pred)
        else:
            return np.array(y_test), np.array(y_pred)


    def tf_convert_predictions_to_labels(y_test, y_pred, class_labels, loss_type="categorical_crossentropy"):
        """
        Converts the ground truth values (y_test) and predicted probabilities (y_pred) into their corresponding class labels.

        Given the true class labels (either one-hot encoded or integer-encoded) and the predicted class probabilities, this function maps each entry to its corresponding class label based on the provided `class_labels` list.

        Parameters:
        -----------
        y_test : numpy.ndarray
            Ground truth labels. If using 'categorical_crossentropy', these should be one-hot encoded. For 'sparse_categorical_crossentropy', these should be integer-encoded.

        y_pred : numpy.ndarray
            Predicted class probabilities, typically from the output of a model's predict method.

        class_labels : list of str
            A list of class labels in which the index of each label corresponds to its class index in predictions and true labels.

        loss_type : str, optional
            Specifies the type of encoding for `y_test` and interpretation for `y_pred`. Supported values are:
            - 'categorical_crossentropy': For one-hot encoded labels.
            - 'sparse_categorical_crossentropy': For integer-encoded labels.
            Defaults to 'categorical_crossentropy'.

        Returns:
        --------
        tuple of list of str
            - The first list contains the true class labels mapped from `y_test`.
            - The second list contains the predicted class labels based on the highest probability in `y_pred`.

        Raises:
        -------
        ValueError:
            If an unsupported `loss_type` value is provided.

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
