import torch
import torch.onnx
import onnx
import ktc
from dataset_preprocessor import DatasetPreprocessor  # Import your preprocessor
import numpy as np

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
preprocessor = DatasetPreprocessor(edge_input_size, inpainting_input_size)

# Preprocess the datasets
edge_img_list = preprocessor.preprocess_edge_model_folder('examples/psv/images', 'examples/psv/masks')

# Add batch dimension to each preprocessed image
edge_img_list_with_batch = [np.expand_dims(img, axis=0) for img in edge_img_list]

# Run analysis
bie_model_path = km.analysis({"input": edge_img_list_with_batch})
print("\nFixed-point analysis done. Save bie model to '" + str(bie_model_path) + "'")

# Compile to NEF model
nef_model_path = ktc.compile([km])
print("\nCompile done. Save Nef file to '" + str(nef_model_path) + "'")
