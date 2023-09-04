import subprocess

from skit.config import IS_PYDICOM_IMPORTED

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
        from enum import Enum
        import numpy as np
        import pandas as pd
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        from skit.Summarizable import Summarizable
        from skit.show import show_text, show_images

    except ImportError:
        print(f"Missing some imports: {ImportError}")

    class ImageFormat(Enum):
        """
        Enum to represent image formats.

        - 'W-H-D-C' = Width-Height-Depth-Channel
        - 'D-W-H-C' = Depth-Width-Height-Channel
        """
        WHDC = "W-H-D-C"  # Format (128, 128, 64, 1)
        DWHC = "D-W-H-C"  # Format (64, 128, 128, 1)

        def swap_dimensions(image, image_format):
            """
            Swap dimensions of an image NumPy array based on the specified permutation type.

            Parameters:
            ----------
            image : numpy.ndarray
                Input image with dimensions to be swapped.
            permutation_type : str
                The category of the scan to load.
                - 'WxHxDxC' = Width-Height-Depth-Channel'
                - 'DxWxHxC' = (Depth-Width-Height-Channel)

            Returns:
            numpy.ndarray: Image with swapped dimensions.
            """
            if image_format == ImageFormat.DWHC:
                # Permutation: (128, 128, 64, 1) vers (64, 128, 128, 1)
                image = np.transpose(image, (2, 1, 0, 3))
            elif image_format == ImageFormat.WHDC:
                # Permutation: (64, 128, 128, 1) vers (128, 128, 64, 1)
                image = np.transpose(image, (2, 1, 0, 3))

            return image

    class DICOMLoader(Summarizable):
        def __init__(self,
                     df,
                     input_path,
                     scan_categories,
                     num_imgs                        = None,
                     size                            = (224, 224),
                     scale                           = 1.0,
                     rotate_angle                    = 0,
                     enable_center_focus             = False,
                     enable_monochrome_normalization = False,
                     id_column_name                  = "ID",
                     label_column_name               = "Label",
                     image_format                    = ImageFormat.WHDC,
                     max_threads                     = 8,
                     image_file_sorter               = lambda x: int(x[:-4].split("-")[-1])
                    ):
            if num_imgs is not None and num_imgs % 2 != 0 and enable_center_focus:
                raise ValueError("num_imgs must be divisible by 2 for central image")

            if not (0 <= rotate_angle <= 360):
                raise ValueError("Rotation value must be between 0 and 360")

            for col in [id_column_name, label_column_name]:
                if col not in df.columns:
                    raise ValueError(f"Columns {col} must be in dataset")

            self.df                  = df
            self.num_imgs            = num_imgs
            self.id_column_name      = id_column_name
            self.label_column_name   = label_column_name
            self.input_path          = input_path
            self.scan_categories     = scan_categories
            self.max_threads         = max_threads
            self.size                = size
            self.scale               = scale
            self.rotate_angle        = rotate_angle
            self.image_format        = image_format
            self.image_file_sorter   = image_file_sorter
            self.enable_center_focus = enable_center_focus

        # ---------------- #
        # Public methods
        # ---------------- #

        # -------- #
        # Info
        # -------- #

        def len(self):
            """
            Returns the length of the DataFrame.

            Returns
            -------
            int
                The number of rows in the DataFrame.
            """
            return len(self.df)


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

        # -------- #
        # Load
        # -------- #

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
                key = self.image_file_sorter
            )

            # Check if any image files were found
            # ----
            if not image_files:
                raise ValueError(f"No image files found in {scans_path}.")

            # Select images based on enable_center_focus and num_imgs
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

            loaded_images = np.array(loaded_images)
            loaded_images = ImageFormat.swap_dimensions(loaded_images, self.image_format)

            return loaded_images

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
            return all_images

        # -------- #
        # Show
        # -------- #

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
            if self.image_format == ImageFormat.WHDC:
                images = ImageFormat.swap_dimensions(images, ImageFormat.DWHC)

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

        # Overriding the summary method
        def summary(self, train_dataset=None):
            super().summary()
            print("Additional summary details specific:")
            # Get the value of self.scan_categories[0]
            # ----
            scan_category = self.scan_categories[0]
            size = self.len()

            if scan_category is not None:
                images = self.load_scan(0, scan_category)
                print("\n")

                if images is not None:
                    print("Size:", size)
                    print("Images Shape:", images.shape)

                else:
                    print("Error: Loading images are empty.")
            else:
                print("Error: scan_category is empty.")
            print("=" * 40)



        # ---------------- #
        # Private methods
        # ---------------- #

        def _load_dicom_image(self, dicom_path):
            """
            Loads a DICOM image from a given path and applies various transformations
            such as VOI LUT, rotation, normalization, cropping, and resizing.

            Parameters
            ----------
            dicom_path : str
                The path to the DICOM file.

            Returns
            -------
            2D array
                The transformed DICOM image.

            Raises
            ------
            FileNotFoundError
                If the specified DICOM file does not exist.
            IOError
                If an error occurs while reading the DICOM file.

            Notes
            -----
            The method performs the following transformations in order:
            1. Applies Value of Interest Lookup Table (VOI LUT) for better visibility.
            2. Rotates the image based on the specified angle.
            3. Normalizes the pixel values in the image.
            4. Crops the image to focus on the region of interest.
            5. Resizes the image to the specified dimensions.
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

            # Get image array
            # ----
            image_array = dicom_file.pixel_array

            # Rotate
            # ----
            image_array = self._rotate_img(image_array)

            # Normalization
            # ----
            image_array = self._normalization_img(image_array)

            # Crop
            # ----
            image_array = self._crop_img(image_array)

            # Resize
            # ----
            image_array = self._resize_img(image_array)

            # Chanel
            # ----
            image_array = np.expand_dims(image_array, axis=-1)

            return image_array

        def _resize_img(self, image_array):
            w, h = self.size

            # Resize
            # ----
            return cv2.resize(image_array, (w, h), interpolation = cv2.INTER_AREA)

        def _crop_img(self, image_array):
            """
            Crops and resizes a given image.

            Parameters
            ----------
            image_array : 2D array
                The image to be cropped and resized.

            Returns
            -------
            2D array
                The cropped and resized image.
            """
            # Calculate the center of the image
            # ----
            center_x, center_y = image_array.shape[1] / 2, image_array.shape[0] / 2

            # Calculate the dimensions of the scaled image
            # ----
            width_scaled, height_scaled = image_array.shape[1] * self.scale, image_array.shape[0] * self.scale

            # Calculate the coordinates for cropping the image
            # ----
            left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
            top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2

            # Crop the image using the calculated coordinates
            # ----
            img_cropped = image_array[int(top_y):int(bottom_y), int(left_x):int(right_x)]

            return img_cropped

        def _rotate_img(self, image_array):
          """
          Rotates the image array by the specified angle.

          Parameters
          ----------
          image_array : ndarray
              The original image array.

          Returns
          -------
          ndarray
              The rotated image array.
          """
          if self.rotate_angle != 0:
            # Math of the rotation matrix
            # ----
            height, width = image_array.shape[:2]
            center = (width / 2, height / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, self.rotate_angle, 1.0)

            # Apply rotation
            # ----
            image_array = cv2.warpAffine(image_array, rotation_matrix, (width, height))

          return image_array


        def _normalization_img(self, image_array):
            """
            Normalizes the image array to a range of 0 to 255.

            Parameters
            ----------
            image_array : ndarray
                The original image array.

            Returns
            -------
            ndarray
                The normalized image array.
            """
            min_val = np.min(image_array)
            max_val = np.max(image_array)

            if max_val == 0:
                # If max value is zero, return an array of zeros
                # ----
                return np.zeros_like(image_array).astype(np.uint8)

            # Otherwise, proceed with normalization
            # ----
            image_array = image_array - min_val
            image_array = image_array / max_val

            return (image_array * 255).astype(np.uint8)

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
            if self.enable_center_focus and self.num_imgs is not None:
                middle = len(image_files) // 2
                num_imgs2 = self.num_imgs // 2
                p1 = max(0, middle - num_imgs2)
                p2 = min(len(image_files), middle + num_imgs2)
                return image_files[p1:p2]

            elif self.num_imgs is not None:
                return image_files[:self.num_imgs]

            else:
                return image_files
