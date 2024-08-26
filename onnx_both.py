import torch
import torch.onnx
import onnx
import ktc
from dataset_preprocessor import DatasetPreprocessor  # Import your preprocessor
import numpy as np

# Load and optimize the ONNX models
exported_edge = onnx.load('PSV_edge_generator.onnx')
result_edge = ktc.onnx_optimizer.torch_exported_onnx_flow(exported_edge)
optimized_edge = ktc.onnx_optimizer.onnx2onnx_flow(result_edge, eliminate_tail=True, opt_matmul=False)
onnx.save(optimized_edge, 'checkpoints/Test/EdgeModel_gen.onnx')

exported_inpainting = onnx.load('PSV_inpaint_generator.onnx')
result_inpainting = ktc.onnx_optimizer.torch_exported_onnx_flow(exported_inpainting)
optimized_inpainting = ktc.onnx_optimizer.onnx2onnx_flow(result_inpainting, eliminate_tail=True, opt_matmul=False)
onnx.save(optimized_inpainting, 'checkpoints/Test/InpaintingModel_gen.onnx')

# Initialize model configurations
km_edge = ktc.ModelConfig(32769, "8b28", "720", onnx_model=optimized_edge)
km_inpainting = ktc.ModelConfig(32770, "8b28", "720", onnx_model=optimized_inpainting)

# Preprocess the datasets
edge_input_size = (256, 256)
inpainting_input_size = (256, 256)
preprocessor = DatasetPreprocessor(edge_input_size, inpainting_input_size)

# Preprocess for Edge model
edge_img_list = preprocessor.preprocess_edge_model_folder('examples/psv/images', 'examples/psv/masks')
edge_img_list_with_batch = []

for img in edge_img_list:
    img_with_batch = np.expand_dims(img, axis=0)  # Shape: (1, 3, 256, 256)
    
    # Add a dummy fourth channel (all zeros)
    dummy_channel = np.zeros((1, 1, 256, 256))  # Shape: (1, 1, 256, 256)
    
    # Concatenate along the channel axis (axis=1) to get 4 channels
    combined_input = np.concatenate((img_with_batch, dummy_channel), axis=1)  # Shape: (1, 4, 256, 256)
    
    edge_img_list_with_batch.append(combined_input)

# Preprocess for Inpainting model
inpainting_img_list = preprocessor.preprocess_inpainting_model_folder('examples/psv/images', 'examples/psv/masks')
inpainting_img_list_with_batch = []

for img, mask in inpainting_img_list:
    img_with_batch = np.expand_dims(img, axis=0)  # Shape: (1, 256, 256, 3)
    mask_with_batch = np.expand_dims(mask, axis=0)  # Shape: (1, 256, 256)
    mask_with_batch = np.expand_dims(mask_with_batch, axis=-1)  # Shape: (1, 256, 256, 1)
    
    combined_input = np.concatenate((img_with_batch, mask_with_batch), axis=-1)  # Shape: (1, 256, 256, 4)
    combined_input = np.transpose(combined_input, (0, 3, 1, 2))  # Shape: (1, 4, 256, 256)
    
    inpainting_img_list_with_batch.append(combined_input)
    
# Run analysis for Inpainting model
bie_model_path_inpainting = km_inpainting.analysis({"input": inpainting_img_list_with_batch})
print("\nFixed-point analysis done. Save bie model to '" + str(bie_model_path_inpainting) + "'")

# Run analysis for Edge model
bie_model_path_edge = km_edge.analysis({"input": edge_img_list_with_batch})
print("\nFixed-point analysis done. Save bie model to '" + str(bie_model_path_edge) + "'")

# Compile both models
batch_compile_result = ktc.compile([km_edge, km_inpainting])
print("\nCompile done. Save Nef file to '" + str(batch_compile_result) + "'")
