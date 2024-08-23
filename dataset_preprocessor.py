import os
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.transform import resize
from PIL import Image

class DatasetPreprocessor:
    def __init__(self, edge_input_size, inpainting_input_size):
        self.edge_input_size = edge_input_size
        self.inpainting_input_size = inpainting_input_size

    def preprocess_edge_model_input(self, image_path, mask_path):
        """
        Preprocess an image and mask for the Edge model.
        """
        # Load the image
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        # Convert to grayscale
        img_gray = rgb2gray(image)

        # Generate edge map using Canny edge detection
        edge_map = canny(img_gray, sigma=2.0).astype(np.float32)

        # Load and preprocess the mask
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask) / 255.0  # Normalize mask to [0, 1]

        # Resize all to the edge model input size
        img_gray_resized = resize(img_gray, self.edge_input_size, preserve_range=True, anti_aliasing=True)
        edge_map_resized = resize(edge_map, self.edge_input_size, preserve_range=True, anti_aliasing=True)
        mask_resized = resize(mask, self.edge_input_size, preserve_range=True, anti_aliasing=True)

        # Stack the grayscale image, edge map, and mask
        stacked_input = np.stack((img_gray_resized, edge_map_resized, mask_resized), axis=0)

        return stacked_input

    def preprocess_inpainting_model_input(self, image_path, mask_path):
        """
        Preprocess an image and mask for the Inpainting model.
        """
        # Load the image
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        # Resize image to the inpainting model input size
        image_resized = resize(image, self.inpainting_input_size, preserve_range=True, anti_aliasing=True)

        # Normalize the image (similar to the original code example)
        image_normalized = image_resized / 256.0 - 0.5

        # Load and preprocess the mask
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask) / 255.0  # Normalize mask to [0, 1]

        # Resize mask to the inpainting model input size
        mask_resized = resize(mask, self.inpainting_input_size, preserve_range=True, anti_aliasing=True)

        return image_normalized, mask_resized

    def preprocess_edge_model_folder(self, image_folder, mask_folder):
        """
        Preprocess all images and masks for the Edge model.
        Returns a list of preprocessed inputs.
        """
        img_list = []
        for dirpath, dirnames, filenames in os.walk(image_folder):
            for f in filenames:
                image_path = os.path.join(dirpath, f)
                mask_path = os.path.join(mask_folder, f)
                processed_input = self.preprocess_edge_model_input(image_path, mask_path)
                img_list.append(processed_input)
        return img_list

    def preprocess_inpainting_model_folder(self, image_folder, mask_folder):
        """
        Preprocess all images and masks for the Inpainting model.
        Returns a list of preprocessed inputs.
        """
        img_list = []
        for dirpath, dirnames, filenames in os.walk(image_folder):
            for f in filenames:
                image_path = os.path.join(dirpath, f)
                mask_path = os.path.join(mask_folder, f)
                image_normalized, mask_resized = self.preprocess_inpainting_model_input(image_path, mask_path)
                img_list.append((image_normalized, mask_resized))
        return img_list
