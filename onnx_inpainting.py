import torch
import torch.onnx
import onnx
import ktc
from dataset_preprocessor import DatasetPreprocessor  # Import your preprocessor
import numpy as np

# Load the ONNX model
exported_inpainting = onnx.load('PSV_inpaint_generator.onnx')

# Optimize the ONNX model
result_inpainting = ktc.onnx_optimizer.torch_exported_onnx_flow(exported_inpainting)
optimized_inpainting = ktc.onnx_optimizer.onnx2onnx_flow(result_inpainting, eliminate_tail=True, opt_matmul=False)

onnx.save(optimized_inpainting, 'checkpoints/Test/InpaintingModel_gen.onnx')

# Initialize the model configuration
km = ktc.ModelConfig(32770, "8b28", "720", onnx_model=optimized_inpainting)

# Set the input sizes
edge_input_size = (256, 256)  # Adjust according to your model
inpainting_input_size = (256, 256)  # Adjust according to your model
preprocessor = DatasetPreprocessor(edge_input_size, inpainting_input_size)

# Add batch dimension to each preprocessed image and mask pair
inpainting_img_list = preprocessor.preprocess_inpainting_model_folder('examples/psv/images', 'examples/psv/masks')

# Combine image and mask into a 4-channel input
inpainting_img_list_with_batch = []
for img, mask in inpainting_img_list:
    # Expand dimensions to add the batch dimension
    img_with_batch = np.expand_dims(img, axis=0)  # Shape: (1, 256, 256, 3)
    mask_with_batch = np.expand_dims(mask, axis=0)  # Shape: (1, 256, 256)
    
    # Expand mask to have a channel dimension and concatenate with image
    mask_with_batch = np.expand_dims(mask_with_batch, axis=-1)  # Shape: (1, 256, 256, 1)
    
    # Concatenate along the channel axis (axis=3)
    combined_input = np.concatenate((img_with_batch, mask_with_batch), axis=-1)  # Shape: (1, 256, 256, 4)
    
    # Rearrange to match the model's expected input shape (batch_size, channels, height, width)
    combined_input = np.transpose(combined_input, (0, 3, 1, 2))  # Shape: (1, 4, 256, 256)
    
    inpainting_img_list_with_batch.append(combined_input)

# Run analysis
bie_model_path = km.analysis({"input": inpainting_img_list_with_batch})
print("\nFixed-point analysis done. Save bie model to '" + str(bie_model_path) + "'")

# Compile to NEF model
nef_model_path = ktc.compile([km])
print("\nCompile done. Save Nef file to '" + str(nef_model_path) + "'")
