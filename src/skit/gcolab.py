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

from skit.config import IN_COLAB

if IN_COLAB:
    """
    This block of code checks if the code is being run in Google Colab.
    If so, it sets up a number of utility functions to handle the Kaggle API.
    """
    try:
        from google.colab import drive
        from skit.utils import mkdir
        from enum import Enum
        import subprocess
        import shutil
        import os
        import glob
        import google.colab


    except ImportError:
        print(f"Missing some imports: {ImportError}")

    class DatasetType(Enum):
        DATASETS = "datasets"
        COMPETITIONS = "competitions"

        def get_flag(self):
            if self == DatasetType.DATASETS:
                return "-d"
            elif self == DatasetType.COMPETITIONS:
                return "-c"
            else:
                return None

    def _install_kaggle_library():
        """
        Installs the Kaggle CLI tool using pip.

        Raises:
        -------
        Exception
            If there's an error during the installation.
        """
        result = subprocess.run(['pip', 'install', 'kaggle'])
        if result.returncode != 0:
          raise Exception("Error on install Kaggle.")

    def gdrive_mount(gdrive_path = '/content/gdrive'):
        """
        Mounts the Google Drive to Colab

        Parameters:
        -----------
        gdrive_path : str
            Path to mount the Google Drive.
        """
        drive.mount(gdrive_path, force_remount=True)

    def _set_environ_kaggle_config(kaggle_config_dir):
        """
        Sets the Kaggle configuration directory.

        Parameters:
        -----------
        kaggle_config_dir : str
            The Kaggle configuration directory located in the Google Drive.
        """
        os.environ['KAGGLE_CONFIG_DIR'] = kaggle_config_dir

    def _is_kaggle_cli_installed():
        """
        Checks if the Kaggle CLI is installed.

        Returns:
        --------
        bool
            True if Kaggle CLI is installed, False otherwise.
        """
        try:
            subprocess.run(['which', 'kaggle'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def _download(kaggle_dataset_url, gdrive_dataset_path, type):
        """
        Downloads and unzips a Kaggle dataset.

        Parameters:
        -----------
        kaggle_dataset_url : str
            The Kaggle dataset URL.

        dataset_destination_dir : str
            The directory where the dataset will be saved and unzipped.

        Raises:
        -------
        Exception
            If the Kaggle CLI is not installed or if there's an error during the download.
        """
        if not _is_kaggle_cli_installed():
          raise Exception("Kaggle CLI is not installed. Please install it using `pip install kaggle`.")

        mkdir(gdrive_dataset_path)
        os.chdir(gdrive_dataset_path)

        try:
          subprocess.run(['kaggle', type.value, 'download', type.get_flag(), kaggle_dataset_url], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        except subprocess.CalledProcessError as e:
          raise Exception(f"An error occurred while downloading the dataset: {e}")

    def download_kaggle_dataset(
        kaggle_dataset_url,
        type                = DatasetType.DATASETS,
        gdrive_path         = '/content/gdrive',
        gdrive_dataset_path = '/content/gdrive/My Drive/dataset',
        kaggle_config_dir   = '/content/gdrive/My Drive/Kaggle'
    ):
        """
        Downloads a Kaggle dataset to a specified Google Drive directory.

        Parameters:
        - kaggle_dataset_url : str
            URL of the Kaggle dataset to download.
        - type : (DatasetType, optional)
            Type of Kaggle dataset (e.g., competitions, datasets).
        - mountpoint_gdrive_path : (str, optional)
            Mount point path for Google Drive.
        - mountpoint_gdrive_dataset_path : (str, optional)
            Path in Google Drive where the dataset will be saved.
        - kaggle_config_dir : (str, optional)
            Directory where Kaggle API credentials are stored.

        Returns:
        - None

        Example usage:
        ```python
        kaggle_dataset_url = "kaggle_url"
        download_kaggle_dataset(
            kaggle_dataset_url,
            DatasetType.COMPETITIONS
        )
        ```

        Note: Make sure to have your Kaggle API credentials in the specified `kaggle_config_dir`.
        """
        try:
            gdrive_mount(gdrive_path)
            _install_kaggle_library()
            _set_environ_kaggle_config(kaggle_config_dir)
            _download(kaggle_dataset_url, gdrive_dataset_path, type)
            print("Dataset downloaded successfully!")

        except Exception as e:
            print(f"An error occurred: {e}")
