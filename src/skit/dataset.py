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

import os
import shutil
import numpy as np
import h5py
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from skit.utils import rmdir
from skit.config import IS_TENSORFLOW_IMPORTED

# ==============================
#           Images
# ==============================

def split_images_dataset(
    data_dir,
    store_dir,
    classes,
    train_ratio=0.7,
    validation_ratio=0.15,
    test_ratio=0.15,
    shuffle_data=True,
    seed=None
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
    if train_ratio + validation_ratio + test_ratio != 1:
        raise ValueError("The sum of the ratios must be 1")

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

    np.random.seed(seed)

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
            if shuffle_data:
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

    np.random.seed(None)

def save_h5_dataset(x_train, y_train, x_test, y_test, x_meta,y_meta, filename):
    """
    Save the datasets into an h5 file.

    Parameters
    ----------
    x_train : numpy array
        Training data features.
    y_train : numpy array
        Training data labels.
    x_test : numpy array
        Test data features.
    y_test : numpy array
        Test data labels.
    x_meta : numpy array
        Metadata of the dataset for training.
    y_meta : numpy array
        Metadata of the dataset for testing.
    filename : str
        The name of the h5 file to be created and saved.
    """
    # Create h5 file
    # ----
    with h5py.File(filename, "w") as f:
        f.create_dataset("x_train", data=x_train)
        f.create_dataset("y_train", data=y_train)
        f.create_dataset("x_test",  data=x_test)
        f.create_dataset("y_test",  data=y_test)
        f.create_dataset("x_meta",  data=x_meta)
        f.create_dataset("y_meta",  data=y_meta)

    # Print
    # ----
    size=os.path.getsize(filename)/(1024*1024)
    print('Dataset : {:24s}  shape : {:22s} size : {:6.1f} Mo   (saved)'.format(filename, str(x_train.shape),size))

def read_dataset(enhanced_dir, dataset_name):
    """
    Read an h5 dataset.

    Parameters
    ----------
    enhanced_dir : str
        Directory containing the h5 dataset.
    dataset_name : str
        Name of the h5 dataset.

    Returns
    -------
    x_train : numpy array
        Training data features.
    y_train : numpy array
        Training data labels.
    x_test : numpy array
        Test data features.
    y_test : numpy array
        Test data labels.
    size : float
        Size of the h5 file in MB.
    """
    # ---- Read dataset
    filename = f'{enhanced_dir}/{dataset_name}.h5'
    size     = os.path.getsize(filename)/(1024*1024)

    with  h5py.File(filename,'r') as f:
        x_train = f['x_train'][:]
        y_train = f['y_train'][:]
        x_test  = f['x_test'][:]
        y_test  = f['y_test'][:]

    # ---- Shuffle
    x_train,y_train=shuffle_np_dataset(x_train,y_train)

    # ---- done
    return x_train,y_train,x_test,y_test,size

def shuffle_np_dataset(*data):
    """
    Shuffle numpy arrays in the same random order.

    Parameters
    ----------
    *data : numpy arrays
        Arrays to be shuffled.

    Returns
    -------
    numpy arrays
        Shuffled arrays in the same random order.
    """
    p = np.random.permutation(len(data[0]))
    out = [ d[p] for d in data ]
    return out[0] if len(out)==1 else out

def rescale_dataset(*data, scale=1):
    """
    Rescale numpy arrays with a scale factor.

    Parameters
    ----------
    *data : numpy arrays
        Arrays to be rescaled.
    scale : float, optional
        Scale factor to adjust the size of the arrays.

    Returns
    -------
    numpy arrays
        Rescaled arrays.
    """
    out = [ d[:int(scale*len(d))] for d in data ]
    return out[0] if len(out)==1 else out

def stratifiedTrainValidSplit(df, x_feature_columns, y_target_columns, num_splits=5, selected_fold=1, seed=None, shuffle=True):
    """
    Splits the DataFrame into training and validation sets using Stratified K-Folds

    NOTE: That resets the index for each train and validation dataframe.

    Parameters:
    -----------
    df : DataFrame
        The DataFrame containing the dataset.
    x_feature_columns : Array of str
        The name of the features column.
    y_target_columns : Array of str
        The name of the labels column.
    num_splits : int
        The number of splits for StratifiedKFold.
    selected_fold : int
        The fold to be used for validation.
    seed : int
        The random seed for reproducibility.
    shuffle : bool
        Whether to shuffle the data before splitting.

    Returns:
    --------
    train_df : DataFrame
        The training set DataFrame.
    valid_df : DataFrame
        The validation set DataFrame.
    """
    # Initialize StratifiedKFold
    # ----
    stratifiedKFold = StratifiedKFold(n_splits=num_splits, random_state=seed, shuffle=shuffle)

    # Add a new column for the fold
    # ----
    df["Fold"] = "train"

    # Prepare the features and labels
    X = df[x_feature_columns]
    y = df[y_target_columns]

    # Perform the split
    for fold_no, (train, valid) in enumerate(stratifiedKFold.split(X, y), start=1):
        if fold_no == selected_fold:
            df.loc[valid, "Fold"] = "valid"

    # Separate into train and valid DataFrames and reset index
    train_df = df[df.Fold == "train"].reset_index(drop=True)
    valid_df = df[df.Fold == "valid"].reset_index(drop=True)

    train_df.drop(columns=['Fold'], inplace=True)
    valid_df.drop(columns=['Fold'], inplace=True)

    return train_df, valid_df

# ==============================
#           TensorFlow
# ==============================

if IS_TENSORFLOW_IMPORTED:
    import tensorflow as tf

    def tf_shuffle_dataset(dataset, batch_size, seed):
        """
        Shuffles a TensorFlow dataset memory-preservingly using a batch-based method and also shuffles the batches themselves.

        Args:
        - dataset :tf.data.Dataset
            The input dataset to shuffle.
        - batch_size : int
            Size of each batch.
        - seed : int, optional
            Seed for shuffle reproducibility.

        Returns:
        - tf.data.Dataset:
            Shuffled dataset.

        Example:
        --------
        Let's consider a dataset: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] and batch_size = 2.

        1. The dataset is divided into the following batches:
           [1, 2], [3, 4], [5, 6], [7, 8], [9, 10]

        2. Each batch is shuffled. Let's assume the shuffled batches are:
           [2, 1], [4, 3], [6, 5], [8, 7], [10, 9] (Note: The actual shuffle might differ)

        3. The order of these shuffled batches is then shuffled. Let's assume the shuffled order is:
           [4, 3], [2, 1], [8, 7], [10, 9], [6, 5] (Note: The actual shuffle might differ)

        4. These batches are concatenated together to give the final shuffled dataset:
           [4, 3, 2, 1, 8, 7, 10, 9, 6, 5]
        """
        if not isinstance(dataset, tf.data.Dataset):
            raise ValueError("The provided dataset is not an instance of tf.data.Dataset.")

        # Split the dataset into batches
        num_elements = sum(1 for _ in dataset)
        num_batches = num_elements // batch_size

        batches = [dataset.skip(i * batch_size).take(batch_size) for i in range(num_batches)]

        # Shuffle each batch individually
        shuffled_batches = [batch.shuffle(buffer_size=batch_size, seed=seed) for batch in batches]

        # Shuffle the order of batches themselves
        batch_order = tf.random.shuffle(tf.range(num_batches), seed=seed)

        # Merge the shuffled batches to create the final dataset
        shuffled_dataset = shuffled_batches[0]
        for i in tqdm(batch_order[1:], desc="Shuffling dataset", unit="batch"):
            shuffled_dataset = shuffled_dataset.concatenate(shuffled_batches[i.numpy()])

        return shuffled_dataset
