import os
import shutil
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartkit.config import IS_TENSORFLOW_IMPORTED

def split_images_dataset(
    data_dir,
    store_dir,
    classes,
    train_ratio=0.7,
    validation_ratio=0.15,
    test_ratio=0.15
):
    """
    Splits images dataset into train, validation and test sets.

    data_dir need to be "MyFolded/ClassA ClassB etc.."

    Parameters
    ----------
    data_dir : str
        Path to the dataset.
    store_dir : str
        Path to store the new dataset Splits.
    classes : list
        List of classes in the dataset.
    train_ratio : float, optional
        Ratio of images to be used for training.
    validation_ratio : float, optional
        Ratio of images to be used for validation.
    test_ratio : float, optional
        Ratio of images to be used for testing.
    """
    # Function to copy files in parallel
    def copy_files(src_path, dest_path, file_names, pbar):
        for file_name in file_names:
            shutil.copy(os.path.join(src_path, file_name), dest_path)
            pbar.update(1)

    # Clean dataset
    if os.path.exists(store_dir):
        rmdir(store_dir)

    total_files = 0
    for cls in classes:
        total_files += len(os.listdir(os.path.join(data_dir, cls)))

    with tqdm(total=total_files, desc="Spliting into train, validation and test sets", unit="file") as pbar:
        for cls in classes:
            # Create directories for train, validation and test for each class
            train_dataset_path = os.path.join(store_dir, 'train', cls)
            validation_dataset_path = os.path.join(store_dir, 'validation', cls)
            test_dataset_path = os.path.join(store_dir, 'test', cls)

            os.makedirs(train_dataset_path)
            os.makedirs(validation_dataset_path)
            os.makedirs(test_dataset_path)

            # Get a list of all files for this class
            src = os.path.join(data_dir, cls)
            all_file_names = os.listdir(src)

            # Shuffle the file names
            np.random.shuffle(all_file_names)

            # Calculate the separation indices for training, validation and testing
            train_idx, val_idx = int(len(all_file_names) * train_ratio), int(len(all_file_names) * (train_ratio + validation_ratio))

            # Split the files between train, validation and test using the indices
            train_file_names = all_file_names[:train_idx]
            val_file_names = all_file_names[train_idx:val_idx]
            test_file_names = all_file_names[val_idx:]

            # Execute copying in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                future_train = executor.submit(copy_files, src, train_dataset_path, train_file_names, pbar)
                future_val = executor.submit(copy_files, src, validation_dataset_path, val_file_names, pbar)
                future_test = executor.submit(copy_files, src, test_dataset_path, test_file_names, pbar)

                # Wait for all copying tasks to complete
                for future in as_completed([future_train, future_val, future_test]):
                    future.result()

if IS_TENSORFLOW_IMPORTED:
    import tensorflow as tf

    def convert_tf_dataset_to_xy(dataset, labels):
        """
        Extracts image data and corresponding labels from a given TensorFlow dataset.

        Parameters:
        dataset : tf.data.Dataset
            The dataset from which image data and labels are to be extracted.
        class_mapping : list
            A list where the index corresponds to the label and the value is the class name.

        Returns:
        tuple : (image_data, class_labels)
            image_data : np.array
                An array of image data.
            class_labels : list
                A list of class names corresponding to the labels in the dataset.
        """
        # Conversion du dataset en une liste d'images et de labels
        x = []
        y = []

        for images, labels_mapping in dataset:
            x.extend(images.numpy())
            y.extend(np.argmax(labels_mapping.numpy(), axis=1))  # get class numbers

        x = np.array(x)
        y = [labels[i] for i in y]  # convert to class names

        return x, y
