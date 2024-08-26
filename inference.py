import cv2
import numpy as np
import kp  # Kneron SDK for KL720
import os
import sys
import argparse
# from dataset_preprocessor import DatasetPreprocessor  # Assuming DatasetPreprocessor is defined in this module
import math
import imageio
from skimage.color import rgb2gray, gray2rgb
from skimage.feature import canny
import torch

EDGE_MODEL_FILE_PATH = 'test_model/models_720.nef'
INPAINTING_MODEL_FILE_PATH = 'InpaintModel/models_720.nef'
IMAGE_FILE_PATH = 'examples/places2/images/places2_01.png'
MASK_FILE_PATH = 'examples/places2/masks/places2_01.png'
LOOP_TIME = 100

def get_device_usb_speed_by_port_id(usb_port_id: int) -> kp.UsbSpeed:
    device_list = kp.core.scan_devices()

    for device_descriptor in device_list.device_descriptor_list:
        if 0 == usb_port_id:
            return device_descriptor.link_speed
        elif usb_port_id == device_descriptor.usb_port_id:
            return device_descriptor.link_speed

    raise IOError('Specified USB port ID {} not exist.'.format(usb_port_id))

def convert_numpy_to_rgba_and_width_align_4(data):
    """Converts the numpy data into RGBA.

    720 input is 4 byte width aligned.

    """

    height, width, channel = data.shape

    width_aligned = 4 * math.ceil(width / 4.0)
    aligned_data = np.zeros((height, width_aligned, 4), dtype=np.int8)
    aligned_data[:height, :width, :channel] = data
    aligned_data = aligned_data.flatten()

    return aligned_data.tobytes()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Edge Connect using KL720.')
    parser.add_argument('-p', '--port_id', help='Using specified port ID for connecting device (Default: port ID of first scanned Kneron device)', default=0, type=int)
    args = parser.parse_args()

    usb_port_id = args.port_id
    
    """
    Check device USB speed (Recommend run KL720 at super speed)
    """
    try:
        if kp.UsbSpeed.KP_USB_SPEED_SUPER != get_device_usb_speed_by_port_id(usb_port_id=usb_port_id):
            print('\033[91m' + '[Warning] Device is not run at super speed.' + '\033[0m')
    except Exception as exception:
        print('Error: check device USB speed fail, port ID = \'{}\', error msg: [{}]'.format(usb_port_id, str(exception)))
        exit(0)
    
    """
    Connect the device
    """
    try:
        print('[Connect Device]')
        device_group = kp.core.connect_devices(usb_port_ids=[usb_port_id])
        print(' - Success')
    except kp.ApiKPException as exception:
        print('Error: connect device fail, port ID = \'{}\', error msg: [{}]'.format(usb_port_id, str(exception)))
        exit(0)        
    """
    Set timeout of the USB communication with the device
    """
    print('[Set Device Timeout]')
    kp.core.set_timeout(device_group=device_group, milliseconds=10000000000000)  # Set to 10 seconds
    print(' - Success')
    
    """
    Upload the Edge Model
    """
    try:
        print('[Upload Edge Model]')
        edge_model_descriptor = kp.core.load_model_from_file(device_group=device_group, file_path=EDGE_MODEL_FILE_PATH)
        print(' - Success')
        
        # nef_radix = edge_model_descriptor.models[0].input_nodes[0].quantization_parameters.v1.quantized_fixed_point_descriptor_list[0].radix
        # print(f'NEF Radix: {nef_radix}')
        
    except kp.ApiKPException as exception:
        print('Error: upload model failed, error = \'{}\''.format(str(exception)))
        exit(0)    
    
    """
    Preprocess the input image and mask using DatasetPreprocessor
    """
    print('[Preprocess Image and Mask for Edge Model]')
    try:
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

        # 確保形狀正確
        input_data = input_data.reshape(1, 3, imgh, imgw)

        print(f'Input Data Shape: {input_data.shape}, dtype: {input_data.dtype}')
        
        # 轉換為字節流
        img_buffer = input_data.tobytes()

        print(' - Preprocessing for Edge Model completed')
    except Exception as e:
        print(f'Error during preprocessing: {str(e)}')
        exit(0)

    """
    Upload and Inference with Edge Model
    """
    try:        
        # Prepare inference descriptor
        inference_descriptor = kp.GenericDataInferenceDescriptor(
            model_id=edge_model_descriptor.models[0].id,
            inference_number=0,
            input_node_data_list=[kp.GenericInputNodeData(buffer=img_buffer)]
        )

        # Debugging: Print the inference descriptor details
        print(f'Inference Descriptor: {inference_descriptor}')
    except kp.ApiKPException as exception:
            print(f'Error: Edge model inference failed, error = \'{exception}\'.')
            exit(0)
        # Send the data for inference
        
    print('[Starting Inference Work]')
    print(' - Starting inference loop {} times'.format(LOOP_TIME))
    print(' - ', end='')
    for i in range(LOOP_TIME):
        try:
            kp.inference.generic_data_inference_send(device_group=device_group,
                                                     generic_inference_input_descriptor=inference_descriptor)

            generic_raw_result = kp.inference.generic_data_inference_receive(device_group=device_group)
        except kp.ApiKPException as exception:
            print(' - Error: inference failed, error = {}'.format(exception))
            exit(0)

        print('.', end='', flush=True)
    print()
    
    """
    retrieve inference node output 
    """
    print('[Retrieve Inference Node Output ]')
    inf_node_output_list = []
    for node_idx in range(generic_raw_result.header.num_output_node):
        inference_float_node_output = kp.inference.generic_inference_retrieve_float_node(
            node_idx=node_idx,
            generic_raw_result=generic_raw_result,
            channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_DEFAULT)
        inf_node_output_list.append(inference_float_node_output)

    print(' - Success')
    # Receive the output from the model
    # edge_output_result = kp.inference.generic_image_inference_receive(device_group=device_group)
    
    # # Convert the output to floating-point data
    # edge_output = kp.inference.generic_inference_retrieve_float_node(
    #     node_idx=0,
    #     generic_raw_result=edge_output_result,
    #     channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_HWCN
    # ).data
    
    print('[Edge Model Inference Completed]')
    

    """
    Preprocess the edge model output and original image for the Inpainting model
    """
    print('[Preprocess Edge Output for Inpainting]')
    try:
        inpainting_input_data = preprocessor.preprocess_inpainting_model(edge_output, 'examples/places2/images/places2_01.png')
        print(' - Preprocessing for Inpainting Model completed')
    except Exception as e:
        print(f'Error during inpainting preprocessing: {str(e)}')
        exit(0)

    """
    Upload and Inference with Inpainting Model
    """
    try:
        print('[Upload Inpainting Model]')
        inpainting_model_descriptor = kp.core.load_model_from_file(device_group=device_group, file_path=INPAINTING_MODEL_FILE_PATH)
        print(' - Success')

        # Prepare inference descriptor
        inference_descriptor = kp.GenericImageInferenceDescriptor(
            model_id=inpainting_model_descriptor.models[0].id,
            inference_number=0,
            input_node_image_list=[
                kp.GenericInputNodeImage(
                    image=inpainting_input_data,
                    image_format=kp.ImageFormat.KP_IMAGE_FORMAT_RGBA8888,  # Adjust according to your model's needs
                    resize_mode=kp.ResizeMode.KP_RESIZE_ENABLE,
                    padding_mode=kp.PaddingMode.KP_PADDING_CORNER,
                    normalize_mode=kp.NormalizeMode.KP_NORMALIZE_KNERON
                )
            ]
        )

        # Send the data for inference
        print('[Run Inference with Inpainting Model]')
        kp.inference.generic_image_inference_send(device_group=device_group, generic_inference_input_descriptor=inference_descriptor)
        
        # Receive the output from the model
        inpainting_output_result = kp.inference.generic_image_inference_receive(device_group=device_group)
        
        # Convert the output to floating-point data
        inpainting_output = kp.inference.generic_inference_retrieve_float_node(
            node_idx=0,
            generic_raw_result=inpainting_output_result,
            channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_HWCN
        ).data
        
        print('[Inpainting Model Inference Completed]')
        
        """
        Postprocess and Save the Output
        """
        output_image_path = 'output_image.jpg'
        inpainting_output = inpainting_output.squeeze()  # Remove unnecessary dimensions
        inpainting_output = np.transpose(inpainting_output, (1, 2, 0))  # Convert to (height, width, channels)
        inpainting_output = (inpainting_output * 255).astype('uint8')  # Convert back to [0, 255] range
        
        cv2.imwrite(output_image_path, inpainting_output)
        print(f'Inpainting result saved to {output_image_path}')

    except kp.ApiKPException as exception:
        print('Error: Inpainting model inference failed, error = \'{}\''.format(str(exception)))
        exit(0)
