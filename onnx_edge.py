import torch
import torch.onnx
import onnx
import ktc
import numpy as np
import imageio
from skimage.color import rgb2gray
from skimage.feature import canny
import cv2
import os

def datasetpreprocessor(IMAGE_FILE_PATH, MASK_FILE_PATH):
    imgh = 256
    imgw = 256
    # Image to Grayscale
    edge_image = imageio.imread(IMAGE_FILE_PATH)
    edge_grayimage = rgb2gray(edge_image)
    edge_grayimage = cv2.resize(edge_grayimage, (imgw, imgh))
    # Mask
    mask = imageio.imread(MASK_FILE_PATH)
    mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
    if len(mask.shape) == 3:  # If the mask is already in RGB format
        mask = rgb2gray(mask)
    mask = (mask > 0).astype(np.uint8) * 255
    # Edge
    tmp_mask = (1-mask/255).astype(bool)
    edge = canny(edge_grayimage, sigma=0, mask=tmp_mask).astype(float)

    # 將數據轉換為 int8 格式
    edge_grayimage = (edge_grayimage * 255).astype(np.int8)
    mask = mask.astype(np.int8)
    edge = (edge * 255).astype(np.int8)

    # 堆疊數據
    input_data = np.stack([edge_grayimage, mask, edge], axis=0)
    input_data = np.expand_dims(input_data, axis=0)

    print(f'Input Data Shape: {input_data.shape}, dtype: {input_data.dtype}')
    
    return input_data

def preprocess_edge_model_folder(image_folder, mask_folder):
        """
        Preprocess all images and masks for the Edge model.
        Returns a list of preprocessed inputs.
        """
        img_list = []
        for dirpath, dirnames, filenames in os.walk(image_folder):
            for f in filenames:
                image_path = os.path.join(dirpath, f)
                mask_path = os.path.join(mask_folder, f)
                processed_input = datasetpreprocessor(image_path, mask_path)   
                img_list.append(processed_input)
        return img_list

# Load the ONNX model
exported_edge = onnx.load('PSV_edge_generator.onnx')

# Optimize the ONNX model
result_edge = ktc.onnx_optimizer.torch_exported_onnx_flow(exported_edge)
optimized_edge = ktc.onnx_optimizer.onnx2onnx_flow(result_edge, eliminate_tail=True, opt_matmul=False)

onnx.save(optimized_edge, 'checkpoints/Test/EdgeModel_gen.onnx')

# Initialize the model configuration
km = ktc.ModelConfig(32769, "8b28", "720", onnx_model=optimized_edge)

# Set the input sizes
edge_input_size = (256, 256)  # Adjust according to your model
inpainting_input_size = (256, 256)  # Adjust according to your model

# Preprocess the datasets
edge_img_list = preprocess_edge_model_folder('examples/psv/images', 'examples/psv/masks')

# Add batch dimension to each preprocessed image
# edge_img_list_with_batch = [np.expand_dims(img, axis=0) for img in edge_img_list]

# Run analysis
bie_model_path = km.analysis({"input": edge_img_list})
print("\nFixed-point analysis done. Save bie model to '" + str(bie_model_path) + "'")

# Compile to NEF model
nef_model_path = ktc.compile([km])
print("\nCompile done. Save Nef file to '" + str(nef_model_path) + "'")
