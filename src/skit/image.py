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
from PIL import Image
from tqdm import tqdm
import concurrent.futures
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def average_image_size(
    dataset_path,
    image_extensions=('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
):
    """
    Calculate the average dimensions (width, height, and channels) of images in a dataset.

    Parameters
    ----------
    dataset_path : str
        The path to the dataset directory.
    image_extensions : tuple, optional
        Tuple of extensions to check.

    Returns : tuple
        A tuple containing the average image dimensions in the form (avg_width, avg_height, avg_channels).
        If no images are found in the dataset, returns (0, 0, 0).
    """
    # Function to get image shape in parallel
    def image_shape(img_path):
        try:
            img = Image.open(img_path)
            if img is not None:
                width, height = img.size
                channels = len(img.getbands())
                return width, height, channels
        except Exception as e:
            print(f"Error processing image at path {img_path}: {str(e)}")
        return None

    avg_dimensions = {'width': [], 'height': [], 'channels': []}
    total_files = 0

    for dirpath, _, filenames in os.walk(dataset_path):
        for filename in filenames:
            if filename.lower().endswith(image_extensions):
                total_files += 1

    with tqdm(total=total_files, desc="Calculate the average images shape", unit="file") as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_img_path = {
                executor.submit(image_shape, os.path.join(dirpath, filename)): os.path.join(dirpath, filename)
                for dirpath, _, filenames in os.walk(dataset_path)
                for filename in filenames
                if filename.lower().endswith(image_extensions)
            }

            for future in concurrent.futures.as_completed(future_to_img_path):
                img_path = future_to_img_path[future]
                result = future.result()
                if result is not None:
                    width, height, channels = result
                    avg_dimensions['width'].append(width)
                    avg_dimensions['height'].append(height)
                    avg_dimensions['channels'].append(channels)
                pbar.update(1)

    # Calculate average dimensions using numpy
    avg_width = np.mean(avg_dimensions['width'])
    avg_height = np.mean(avg_dimensions['height'])
    avg_channels = np.mean(avg_dimensions['channels'])

    return avg_width, avg_height, avg_channels
