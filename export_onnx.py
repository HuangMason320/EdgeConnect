import torch
import torch.onnx
from src.models import EdgeGenerator, InpaintGenerator

def load_state_dict(model, state_dict):
    # 創建一個新的 OrderedDict，只包含模型結構中存在的鍵
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('generator.'):
            name = k[10:]  # 移除 "generator." 前綴
        else:
            name = k
        if name in model.state_dict():
            new_state_dict[name] = v
    
    # 加載新的 state dict
    model.load_state_dict(new_state_dict, strict=False)
    return model

def convert_edge_generator_to_onnx(pth_path, onnx_path, input_shape):
    # 加載 EdgeGenerator
    model = EdgeGenerator(use_spectral_norm=True)
    state_dict = torch.load(pth_path, map_location='cpu')
    if 'generator' in state_dict:
        state_dict = state_dict['generator']
    model = load_state_dict(model, state_dict)
    model.eval()

    # 創建示例輸入 (grayscale + edge + mask)
    dummy_input = torch.randn(input_shape)

    # 導出 ONNX 模型
    torch.onnx.export(model, 
                      dummy_input,
                      onnx_path,
                      export_params=True,
                      opset_version=18,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes=None)

    print(f"Edge Generator 模型已成功轉換為 ONNX 格式並保存至 {onnx_path}")

def convert_inpaint_generator_to_onnx(pth_path, onnx_path, input_shape):
    # 加載 InpaintGenerator
    model = InpaintGenerator()
    state_dict = torch.load(pth_path, map_location='cpu')
    if 'generator' in state_dict:
        state_dict = state_dict['generator']
    model = load_state_dict(model, state_dict)
    model.eval()

    # 創建示例輸入 (rgb + edge)
    dummy_input = torch.randn(input_shape)

    # 導出 ONNX 模型
    torch.onnx.export(model, 
                      dummy_input,
                      onnx_path,
                      export_params=True,
                      opset_version=18,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes=None)

    print(f"Inpaint Generator 模型已成功轉換為 ONNX 格式並保存至 {onnx_path}")

# 使用示例
if __name__ == "__main__":
    # PSV
    # Edge Generator
    convert_edge_generator_to_onnx(
        pth_path="checkpoints/ParisStreetview/EdgeModel_gen.pth",
        onnx_path="PSV_edge_generator.onnx",
        input_shape=(1, 3, 256, 256)  # 批次大小為1，3通道 (grayscale + edge + mask)，512x512圖像
    )

    # Inpaint Generator
    convert_inpaint_generator_to_onnx(
        pth_path="checkpoints/ParisStreetview/InpaintingModel_gen.pth",
        onnx_path="PSV_inpaint_generator.onnx",
        input_shape=(1, 4, 256, 256)  # 批次大小為1，4通道 (rgb + edge)，512x512圖像
    )
    
    # Places2 
    convert_edge_generator_to_onnx(
        pth_path="checkpoints/Places2/EdgeModel_gen.pth",
        onnx_path="Places2_edge_generator.onnx",
        input_shape=(1, 3, 256, 256)  # 批次大小為1，3通道 (grayscale + edge + mask)，512x512圖像
    )

    # Inpaint Generator
    convert_inpaint_generator_to_onnx(
        pth_path="checkpoints/Places2/InpaintingModel_gen.pth",
        onnx_path="Place2_inpaint_generator.onnx",
        input_shape=(1, 4, 256, 256)  # 批次大小為1，4通道 (rgb + edge)，512x512圖像
    )