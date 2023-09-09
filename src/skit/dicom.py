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
        from skit.InternalDebug import InternalDebug

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
            id_column_name                  = "ID",
            label_column_name               = "Label",
            image_format                    = ImageFormat.WHDC,
            max_threads                     = 8,
            image_file_sorter               = lambda x: int(x[:-4].split("-")[-1]),
            debug_mode                      = False
        ):
            """
            This class is designed for loading DICOM images from a given directory and
            creating a dataset for medical image analysis.

            Parameters:
            -----------
            df : DataFrame
                The DataFrame containing metadata and labels for DICOM images.
            input_path : str
                The path to the directory containing DICOM image files.
            scan_categories : list
                A list of scan categories to include in the dataset.
            num_imgs : int, optional
                The number of images to load per scan. If specified, must be divisible
                by 2 when 'enable_center_focus' is True. Default is None.
            size : tuple, optional
                The size to which images should be resized. Default is (224, 224).
            scale : float, optional
                The scaling factor applied to the images. Default is 1.0.
            rotate_angle : int, optional
                The angle in degrees by which images should be rotated. Default is 0.
            enable_center_focus : bool, optional
                If True, focus on the central images when 'num_imgs' is specified.
                Default is False.
            id_column_name : str, optional
                The name of the column containing unique IDs in 'df'. Default is "ID".
            label_column_name : str, optional
                The name of the column containing labels in 'df'. Default is "Label".
            image_format : ImageFormat, optional
                The format of the DICOM images (e.g., ImageFormat.WHDC). Default is
                ImageFormat.WHDC.
            max_threads : int, optional
                The maximum number of threads to use for image loading. Default is 8.
            image_file_sorter : function, optional
                A function used to sort image files. Default sorts by numeric value
                at the end of the filename.
            debug_mode: bool
                Debug mode

            Raises:
            -------
            ValueError
                - If 'num_imgs' is not divisible by 2 when 'enable_center_focus' is True.
                - If 'rotate_angle' is not in the range [0, 360].
                - If 'id_column_name' or 'label_column_name' is not in 'df.columns'.
            """
            if num_imgs is not None and num_imgs % 2 != 0 and enable_center_focus:
                raise ValueError("num_imgs must be divisible by 2 for central image")

            if not (0 <= rotate_angle <= 360):
                raise ValueError("Rotation value must be between 0 and 360")

            for col in [id_column_name, label_column_name]:
                if col not in df.columns:
                    raise ValueError(f"Columns {col} must be in dataset")

            self.__df                  = df
            self.__num_imgs            = num_imgs
            self.__id_column_name      = id_column_name
            self.__label_column_name   = label_column_name
            self.__input_path          = input_path
            self.__scan_categories     = scan_categories
            self.__max_threads         = max_threads
            self.__size                = size
            self.__scale               = scale
            self.__rotate_angle        = rotate_angle
            self.__image_format        = image_format
            self.__image_file_sorter   = image_file_sorter
            self.__enable_center_focus = enable_center_focus
            self.__debug               = InternalDebug(debug_mode=debug_mode)

        # ---------------- #
        # Public
        # ---------------- #

        # -------- #
        # Property
        # -------- #

        @property
        def scan_categories(self):
            return self.__scan_categories

        @property
        def num_imgs(self):
            return self.__num_imgs

        @property
        def image_format(self):
            return self.__image_format

        @property
        def df(self):
            return self.__df.copy()

        @property
        def len(self):
            """
            Returns the length of the DataFrame.

            Returns
            -------
            int
                The number of rows in the DataFrame.
            """
            return len(self.__df)

        # -------- #
        # Get
        # -------- #

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
            return self.__df.loc[row, self.__id_column_name]

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
            return self.__df.loc[row, self.__label_column_name]

        # -------- #
        # Format
        # -------- #

        def format(self, images, type):
            if type == "normalize" and self.image_format == ImageFormat.WHDC:
                return ImageFormat.swap_dimensions(images, ImageFormat.DWHC)

            elif type == "default" and self.image_format == ImageFormat.WHDC:
                return ImageFormat.swap_dimensions(images, ImageFormat.WHDC)

            else:
                return images

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
            self.__debug.log("== load_scan ==")
            # Get the patient ID and construct the scan path
            # ----
            patient_id = str(self.__df.loc[row, self.__id_column_name]).zfill(5)
            scans_path = os.path.join(self.__input_path, patient_id, scan_category)

            # Check if the scan path exists
            # ----
            if not os.path.exists(scans_path):
                raise FileNotFoundError(f"The folder {scans_path} doesn't exist.")

            # Get the list of image files
            # ----
            image_files = sorted(
                glob.glob(os.path.join(scans_path, "*")),
                key = self.__image_file_sorter
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
            with ThreadPoolExecutor(self.__max_threads) as executor:
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
            if self.__num_imgs is not None:
                while len(loaded_images) < self.__num_imgs:
                    zero_image = np.zeros_like(loaded_images[0])
                    loaded_images.append(zero_image)

            loaded_images = np.array(loaded_images)

            return self.format(loaded_images, "default")

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
            self.__debug.log("== load_all_scans ==")
            # Initialize an empty dictionary to store images for each MRI type
            # ----
            all_images = {}

            # Use a ThreadPoolExecutor to load images for each MRI type concurrently
            # ----
            with ThreadPoolExecutor(self.__max_threads) as executor:
                future_to_scan_category = {
                    executor.submit(
                        self.load_scan,
                        row,
                        scan_category,
                        False
                    ): scan_category for scan_category in self.__scan_categories
                }

                # Initialize a tqdm progress bar
                # ----
                if show_progress:
                    progress_bar = tqdm(total=len(self.__scan_categories), desc="Loading scan types")

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
            return {key: all_images.get(key, []) for key in self.__scan_categories}

        # -------- #
        # Show
        # -------- #

        def _show_scan(self, scan_category, images, color_map):
            images = self.format(images, "normalize")
            show_text("h4", scan_category, False)
            show_images(images, color_map=color_map)

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
            self._show_scan(scan_category, images, color_map)


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
            loaders_images = self.load_all_scans(row)
            for scan_category, images in loaders_images.items():
                self._show_scan(scan_category, images, color_map)

        # Overriding the summary method
        def summary(self, train_dataset=None):
            super().summary()
            print("Additional summary details specific:")
            # Get the value of self.__scan_categories[0]
            # ----
            scan_category = self.__scan_categories[0]

            if scan_category is not None:
                images = self.load_scan(0, scan_category)
                print("\n")

                if images is not None:
                    print("Size:", self.len)
                    print("Images Shape:", images.shape)

                else:
                    print("Error: Loading images are empty.")
            else:
                print("Error: scan_category is empty.")
            print("=" * 50)

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
            self.__debug.log("== _load_dicom_image ==")
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
            image = dicom_file.pixel_array
            self.__debug.log("Pixel array shape:", image.shape)

            # Rotate
            # ----
            image = self._rotate_img(image)

            # Normalization
            # ----
            image = self._normalization_img(image)

            # Crop
            # ----
            image = self._crop_img(image)

            # Resize
            # ----
            image = self._resize_img(image)

            # Chanel
            # ----
            image = np.expand_dims(image, axis=-1)

            self.__debug.log("Is normalized: ", self._is_normalized(image))
            self.__debug.log("Image shape: ", image.shape)

            return image

        def _resize_img(self, image):
            w, h = self.__size

            # Resize
            # ----
            return cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)

        def _crop_img(self, image):
            """
            Crops and resizes a given image.

            Parameters
            ----------
            image : 2D array
                The image to be cropped and resized.

            Returns
            -------
            2D array
                The cropped and resized image.
            """
            # Skip if no crop
            # ----
            if self.__scale <= 0:
                return image

            # Calculate the center of the image
            # ----
            center_x, center_y = image.shape[1] / 2, image.shape[0] / 2

            # Calculate the dimensions of the scaled image
            # ----
            width_scaled, height_scaled = image.shape[1] * self.__scale, image.shape[0] * self.__scale

            # Calculate the coordinates for cropping the image
            # ----
            left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
            top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2

            # Crop the image using the calculated coordinates
            # ----
            return image[int(top_y):int(bottom_y), int(left_x):int(right_x)]

        def _rotate_img(self, image):
          """
          Rotates the image array by the specified angle.

          Parameters
          ----------
          image : ndarray
              The original image array.

          Returns
          -------
          ndarray
              The rotated image array.
          """
          if self.__rotate_angle <= 0:
              return image

          #    Math of the rotation matrix
          # ----
          height, width = image.shape[:2]
          center = (width / 2, height / 2)
          rotation_matrix = cv2.getRotationMatrix2D(center, self.__rotate_angle, 1.0)

          # Apply rotation
          # ----
          return cv2.warpAffine(image, rotation_matrix, (width, height))

        def _normalization_img(self, image):
            """
            Normalizes the image array to a range of 0 to 255.

            Parameters
            ----------
            image : ndarray
                The original image array.

            Returns
            -------
            ndarray
                The normalized image array.
            """
            min_val = np.min(image)
            max_val = np.max(image)

            if max_val == 0:
                # If max value is zero, return an array of zeros
                # ----
                return np.zeros_like(image).astype(np.uint8)

            # Otherwise, proceed with normalization
            # ----
            image = image - min_val
            image = image / max_val

            return (image * 255).astype(np.uint8)

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
            if self.__enable_center_focus and self.__num_imgs is not None:
                middle = len(image_files) // 2
                num_imgs2 = self.__num_imgs // 2
                p1 = max(0, middle - num_imgs2)
                p2 = min(len(image_files), middle + num_imgs2)
                return image_files[p1:p2]

            elif self.__num_imgs is not None:
                return image_files[:self.__num_imgs]

            else:
                return image_files


        def _is_normalized(self, image):
            min_value = np.min(image)
            max_value = np.max(image)

            return min_value >= 0.0 and max_value <= 255
