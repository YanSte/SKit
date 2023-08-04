from skit.config import IN_COLAB
from skit.utils import mkdir

if IN_COLAB:
    try:
        import subprocess
        import shutil
        import os
        import glob
        import google.colab
        from google.colab import drive

    except ImportError:
        print(f"Missing some imports: {ImportError}")

    def install_kaggle():
        result = subprocess.run(['pip', 'install', 'kaggle'])
        if result.returncode != 0:
          raise Exception("Error on install Kaggle.")

    def set_environ_kaggle_config(
        mountpoint_gdrive_path,
        kaggle_config_dir
    ):
        drive.mount(f'{mountpoint_gdrive_path}/gdrive', force_remount=True)
        os.environ['KAGGLE_CONFIG_DIR'] = f"{mountpoint_gdrive_path}/gdrive/My Drive/{kaggle_config_dir}"

    def is_kaggle_cli_installed():
      try:
          subprocess.run(['which', 'kaggle'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
          return True
      except subprocess.CalledProcessError:
          return False

    def download_and_unzip_dataset(kaggle_dataset_url, dataset_destination_dir):
        if not is_kaggle_cli_installed():
          raise Exception("Kaggle CLI is not installed. Please install it using `pip install kaggle`.")

        mkdir(dataset_destination_dir)
        os.chdir(dataset_destination_dir)

        try:
          subprocess.run(['kaggle', 'datasets', 'download', '-d', kaggle_dataset_url], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
          zip_files = glob.glob("*.zip")

          # Unzip each ZIP file one by one
          for zip_file in zip_files:
            subprocess.run(['unzip', zip_file])
            os.remove(zip_file)

        except subprocess.CalledProcessError as e:
          raise Exception(f"An error occurred while downloading the dataset: {e}")

    def setup_kaggle_dataset(
        kaggle_dataset_url,
        dataset_destination_path = '/content',
        mountpoint_gdrive_path = '/content',
        kaggle_config_dir = 'Kaggle'
    ):
        try:
            install_kaggle()
            set_environ_kaggle_config(mountpoint_gdrive_path, kaggle_config_dir)
            download_and_unzip_dataset(kaggle_dataset_url, dataset_destination_path)
            print("Dataset downloaded and unzipped successfully!")

        except Exception as e:
            print(f"An error occurred: {e}")
