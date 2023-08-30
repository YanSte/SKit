import subprocess

def unzip_specific_files(zip_file_path, destination_directory, specific_files=None):
    """
    Unzips specific files or folders from a ZIP archive.

    Parameters:
    - zip_file_path (str): Path to the ZIP file to unzip.
    - destination_directory (str): Directory where the files will be extracted.
    - specific_files (list, optional): List of specific files or folders to extract.

    Example usage:
    ```python
    zip_file_path = "/path/to/archive.zip"
    destination_directory = "/path/to/destination/"
    specific_files = ["file1", "file2", "folder1/"]
    unzip_specific_files(zip_file_path, destination_directory, specific_files)
    ```
    """
    # Construire la commande unzip
    unzip_command = ["unzip", zip_file_path, "-d", destination_directory]

    # Ajouter des fichiers ou dossiers spécifiques à la commande, si fournis
    if specific_files:
        unzip_command[2:2] = specific_files

    # Exécuter la commande unzip
    subprocess.run(unzip_command)
