from skit.config import IS_PYDICOM_IMPORTED
import subprocess

def install_pydicom_library():
        result = subprocess.run(['pip', 'install', '-q','pydicom'])
        if result.returncode == 0:
            print("Pydicom installed 📦")
        else:
          raise Exception("Error on install Kaggle.")

if IS_PYDICOM_IMPORTED:
    """
    This block of code checks if the code is being run with PYDICOM Lib.
    """
    try:
        import pydicom
        import os
        import glob
        import cv2
        import numpy as np
        import pandas as pd
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        from skit.Summarizable import Summarizable
        from skit.show import show_text, show_images

    except ImportError:
        print(f"Missing some imports: {ImportError}")

    class DICOMLoader(Summarizable):
        def __init__(self,
                     df,
                     input_path,
                     scan_categories,
                     num_imgs          = None,
                     central_focus     = False,
                     scale_dim         = (1.0, (244,244)),
                     rotate            = None,
                     id_column_name    = "ID",
                     label_column_name = "Label",
                     max_threads       = 8
                    ):
            """
            Initializes the DICOMLoader object with various parameters.

            Parameters
            ----------
            df : DataFrame
                The DataFrame containing the dataset.
            input_path : str
                The path to the DICOM files.
            scan_categories : list
                List of scan categories to be loaded.
            num_imgs : int, optional
                Number of images to load for each scan.
            central_focus : bool, optional
                Whether to focus on the central part of the image.
            rotate : int, optional
                Rotation angle for the image. `0:0, 1:ROTATE_90_CLOCKWISE, 2:ROTATE_90_COUNTERCLOCKWISE, 3:ROTATE_180`
            id_column_name : str, optional
                Column name for the ID in the DataFrame.
            label_column_name : str, optional
                Column name for the label in the DataFrame.
            scale_dim : tuple, optional
                Tuple containing the scale and dimensions for image resizing.
            max_threads : int, optional
                Maximum number of threads for parallel execution.
            """
            if num_imgs is not None and num_imgs % 2 != 0 and central_focus:
                raise ValueError("num_imgs must be divisible by 2 for central image")

            for col in [id_column_name, label_column_name]:
                if col not in df.columns:
                    raise ValueError(f"Columns {col} must be in dataset")

            self.df                = df
            self.num_imgs          = num_imgs
            self.central_focus     = central_focus
            self.id_column_name    = id_column_name
            self.label_column_name = label_column_name
            self.input_path        = input_path
            self.scan_categories   = scan_categories
            self.max_threads       = max_threads
            self.scale_dim         = scale_dim
            self.rotate            = rotate

        def len(self):
            """
            Returns the length of the DataFrame.

            Returns
            -------
            int
                The number of rows in the DataFrame.
            """
            return len(self.df)

        def _crop_resize_img(self, img):
            """
            Crops and resizes a given image.

            Parameters
            ----------
            img : 2D array
                The image to be cropped and resized.

            Returns
            -------
            2D array
                The cropped and resized image.
            """
            # Retrieve the scaling factor and dimensions from the object's attributes
            # ----
            scale, dim = self.scale_dim

            # Calculate the center of the image
            # ----
            center_x, center_y = img.shape[1] / 2, img.shape[0] / 2

            # Calculate the dimensions of the scaled image
            # ----
            width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale

            # Calculate the coordinates for cropping the image
            # ----
            left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
            top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2

            # Crop the image using the calculated coordinates
            # ----
            img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]

            # Resize
            # ----
            img_cropped = cv2.resize(img_cropped, dim, interpolation = cv2.INTER_AREA)

            return img_cropped

        def _load_dicom_image(self, dicom_path):
            """
            Loads and normalizes a DICOM image from a given path.

            Parameters
            ----------
            dicom_path : str
                The path to the DICOM file.

            Returns
            -------
            2D array
                The loaded and normalized DICOM image.
            """
            # Check if the DICOM file exists
            # ----
            if not os.path.exists(dicom_path):
                raise FileNotFoundError(f"File {dicom_path} does not exist.")

            # Load the DICOM file
            # ----
            try:
                dicom_file = pydicom.dcmread(dicom_path)
            except Exception as e:
                raise IOError(f"An error occurred while reading the DICOM file: {e}")

            # Extract the pixel array from the DICOM file
            # ----
            image_array = dicom_file.pixel_array

            if self.rotate is not None:
                rot_choices = [0, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]
                image_array = cv2.rotate(image_array, rot_choices[self.rotate])

            # Calculate min and max values in one go
            # ----
            min_val, max_val = np.min(image_array), np.max(image_array)

            # Check if normalization is necessary
            # ----
            if min_val != max_val:
                if max_val == 0:
                    raise ValueError("The maximum value of the image is zero, normalization will fail.")

                # Perform normalization
                # ----
                normalized_image = (image_array - min_val) / (max_val - min_val) * 255
                normalized_image = normalized_image.astype(np.uint8)
            else:
                normalized_image = image_array

            # Crop and resize the image
            # ----
            cropped_resized_img = self._crop_resize_img(normalized_image)

            return cropped_resized_img

        def _select_subset_image_files(self, image_files):
            """
            Selects a subset of image files based on the object's attributes.

            Parameters
            ----------
            image_files : list
                List of image files to select from.

            Returns
            -------
            list
                A subset of the original list of image files.
            """
            if self.central_focus and self.num_imgs is not None:
                middle = len(image_files) // 2
                num_imgs2 = self.num_imgs // 2
                p1 = max(0, middle - num_imgs2)
                p2 = min(len(image_files), middle + num_imgs2)
                return image_files[p1:p2]

            elif self.num_imgs is not None:
                return image_files[:self.num_imgs]

            else:
                return image_files


        def load_scan(self, row, scan_category, show_progress=True):
            """
            Loads a scan for a given row and scan category.

            Parameters
            ----------
            row : int
                The row index in the DataFrame.
            scan_category : str
                The category of the scan to load.
            show_progress : bool, optional
                Whether to show a progress bar.

            Returns
            -------
            list
                A list of loaded images.
            """
            # Get the patient ID and construct the scan path
            # ----
            patient_id = str(self.df.loc[row, self.id_column_name]).zfill(5)
            scans_path = os.path.join(self.input_path, patient_id, scan_category)

            # Check if the scan path exists
            # ----
            if not os.path.exists(scans_path):
                raise FileNotFoundError(f"The folder {scans_path} doesn't exist.")

            # Get the list of image files
            # ----
            image_files = sorted(
                glob.glob(os.path.join(scans_path, "*")),
                key=lambda x: int(x[:-4].split("-")[-1])
            )

            # Check if any image files were found
            # ----
            if not image_files:
                raise ValueError(f"No image files found in {scans_path}.")

            # Select images based on central_focus and num_imgs
            # ----
            image_files = self._select_subset_image_files(image_files)

            # Initialize the list to store loaded images
            # ----
            loaded_images = []

            # Load images using ThreadPoolExecutor
            # ----
            with ThreadPoolExecutor(self.max_threads) as executor:
                image_data_iterable = executor.map(self._load_dicom_image, image_files)

                # Show progress if enabled
                # ----
                if show_progress:
                    image_data_iterable = tqdm(image_data_iterable, total=len(image_files), desc="Loading images")

                # Append loaded images to the list
                # ----
                for image_data in image_data_iterable:
                    if image_data is not None:
                        loaded_images.append(image_data)

            # Check if any images were loaded
            # ----
            if not loaded_images:
                raise ValueError("No images were loaded, and num_imgs is set. Cannot proceed.")

            # Fill up the list to num_imgs if necessary
            # ----
            if self.num_imgs is not None:
                while len(loaded_images) < self.num_imgs:
                    zero_image = np.zeros_like(loaded_images[0])
                    loaded_images.append(zero_image)

            return loaded_images


        def get_id(self, row):
            """
            Retrieves the ID for a given row in the DataFrame.

            Parameters
            ----------
            row : int
                The row index in the DataFrame.

            Returns
            -------
            str
                The ID corresponding to the row.
            """
            return self.df.loc[row, self.id_column_name]

        def gel_label(self, row):
            """
            Retrieves the label for a given row in the DataFrame.

            Parameters
            ----------
            row : int
                The row index in the DataFrame.

            Returns
            -------
            str
                The label corresponding to the row.
            """
            return self.df.loc[row, self.label_column_name]

        def load_all_scans(
            self,
            row,
            show_progress=True
        ):
            """
            Loads all scans for a given row in the DataFrame.

            Parameters
            ----------
            row : int
                The row index in the DataFrame.
            show_progress : bool, optional
                Whether to show a progress bar.

            Returns
            -------
            dict
                A dictionary containing all loaded images, categorized by scan type.
            """
            # Initialize an empty dictionary to store images for each MRI type
            # ----
            all_images = {}

            # Use a ThreadPoolExecutor to load images for each MRI type concurrently
            # ----
            with ThreadPoolExecutor(self.max_threads) as executor:
                future_to_scan_category = {
                    executor.submit(
                        self.load_scan,
                        row,
                        scan_category,
                        False
                    ): scan_category for scan_category in self.scan_categories
                }

                # Initialize a tqdm progress bar
                # ----
                if show_progress:
                    progress_bar = tqdm(total=len(self.scan_categories), desc="Loading scan types")

                # As each task completes, store the loaded images in the all_images dictionary and update the progress bar
                # ----
                for future in as_completed(future_to_scan_category):
                    scan_category = future_to_scan_category[future]
                    image_data = future.result()

                    if image_data is not None:
                        all_images[scan_category] = image_data
                        if show_progress:
                            progress_bar.update(1)

                # Close the progress bar if it exists
                # ----
                if show_progress:
                    progress_bar.close()

            # Ordering by categories
            # ----
            all_images = {key: all_images.get(key, []) for key in self.scan_categories}
            return  all_images

        def show(self, row, scan_category, color_map='gray'):
            """
            Displays the images for a given row and scan category.

            Parameters
            ----------
            row : int
                The row index in the DataFrame.
            scan_category : str
                The category of the scan to display.
            color_map : str, optional
                The color map to use for displaying the images.

            """
            images = self.load_scan(row, scan_category)
            show_text("h4", scan_category)
            show_images(images, color_map=color_map)

        def show_all(self, row, color_map='gray'):
            """
            Displays all images for a given row in the DataFrame.

            Parameters
            ----------
            row : int
                The row index in the DataFrame.
            color_map : str, optional
                The color map to use for displaying the images.

            """
            for scan_category in self.scan_categories:
                self.show(row, scan_category, color_map)
