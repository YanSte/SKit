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

    def install_kaggle():
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

    def gdrive_mount(mountpoint_gdrive_path):
        """
        Mounts the Google Drive to Colab

        Parameters:
        -----------
        mountpoint_gdrive_path : str
            Path to mount the Google Drive.
        """
        drive.mount(mountpoint_gdrive_path, force_remount=True)

    def set_environ_kaggle_config(kaggle_config_dir):
        """
        Sets the Kaggle configuration directory.

        Parameters:
        -----------
        kaggle_config_dir : str
            The Kaggle configuration directory located in the Google Drive.
        """
        os.environ['KAGGLE_CONFIG_DIR'] = kaggle_config_dir

    def is_kaggle_cli_installed():
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

    def download_and_unzip_dataset(kaggle_dataset_url, dataset_destination_dir, type):
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
        if not is_kaggle_cli_installed():
          raise Exception("Kaggle CLI is not installed. Please install it using `pip install kaggle`.")

        mkdir(dataset_destination_dir)
        os.chdir(dataset_destination_dir)

        try:
          subprocess.run(['kaggle', type.value, 'download', type.get_flag(), kaggle_dataset_url], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
          zip_files = glob.glob("*.zip")

          # Unzip each ZIP file one by one
          for zip_file in zip_files:
            subprocess.run(['unzip', zip_file])
            os.remove(zip_file)

        except subprocess.CalledProcessError as e:
          raise Exception(f"An error occurred while downloading the dataset: {e}")

    def setup_kaggle_dataset(
        kaggle_dataset_url,
        type                     = DatasetType.DATASETS,
        dataset_destination_path = '/content',
        mountpoint_gdrive_path   = '/content/gdrive',
        kaggle_config_dir        = '/content/gdrive/My Drive/Kaggle'
    ):
        """
        Sets up a Kaggle dataset in Google Colab by installing required tools,
        setting configurations, downloading, and unzipping the dataset.

        Parameters:
        -----------
        kaggle_dataset_url : str
            The Kaggle dataset URL.

        type : str
            Type of Datasets

        dataset_destination_path : str, optional
            The directory where the dataset will be saved and unzipped.
            Default is '/content'.

        mountpoint_gdrive_path : str, optional
            Path to mount the Google Drive.
            Default is '/content'.

        kaggle_config_dir : str, optional
            The Kaggle configuration directory located in the Google Drive.
            Default is 'Kaggle'.
        """
        try:
            gdrive_mount(mountpoint_gdrive_path)

            if os.path.exists(dataset_destination_path) == False:
                install_kaggle()
                set_environ_kaggle_config(kaggle_config_dir)
                download_and_unzip_dataset(kaggle_dataset_url, dataset_destination_path, type)
                print("Dataset downloaded and unzipped successfully!")
            else:
                print(f"Dataset already exists at {dataset_destination_path}. Skipping download.")
                return

        except Exception as e:
            print(f"An error occurred: {e}")
