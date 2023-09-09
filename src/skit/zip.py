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

import subprocess

def unzip_specific_files(
    zip_file_path,
    destination_directory,
    specific_files=None
):
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
    specific_files = [
        "file1.txt",
        "file2.csv",
        "folder1/*"
    ]

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
