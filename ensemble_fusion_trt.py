import torch
import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import utils
import matplotlib.pyplot as plt
import time
from PIL import Image
from args_fusion import get_parser



TRT_LOGGER = trt.Logger()

# Filenames of TensorRT plan file and input/output images.
decoder_engine = "resccfusion_decoder"
encoder_engine = "resccfusion_encoder"
fusion_onnx = "resccfusion_fusion"
combined_engine = "rescbamfusion"

def load_engine(engine_file_name):
    path = f'E:\\Projects\\THERMAL_MONOCULAR\\ResCCFusion-Fork\\{engine_file_name}.trt'
    assert os.path.exists(path)
    print("Reading engine from file {}".format(path))
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
    
def encode_infer(engine, input_images, image_height, image_width):
    with engine.create_execution_context() as context:
        # Allocate host and device buffers
        tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        for tensor in tensor_names:
            size = trt.volume(context.get_tensor_shape(tensor))
            dtype = trt.nptype(engine.get_tensor_dtype(tensor))
            
            if engine.get_tensor_mode(tensor) == trt.TensorIOMode.INPUT:
                context.set_input_shape(tensor, (2, 1, image_height, image_width))
                input_buffer = np.ascontiguousarray(input_images)
                input_memory = cuda.mem_alloc(input_images.nbytes)
                context.set_tensor_address(tensor, int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                context.set_tensor_address(tensor, int(output_memory))

        stream = cuda.Stream()
        
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        
        # Run inference
        start = time.time()
        context.execute_async_v3(stream_handle=stream.handle)
        end = time.time()
        print("Encode inference time:",(end - start))
        # Transfer prediction output from the GPU.
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        
        # Synchronize the stream
        stream.synchronize()

        return np.reshape(output_buffer, (2,112, image_height, image_width))

def fusion_infer(engine, input, image_height, image_width):
    with engine.create_execution_context() as context:
        # Allocate host and device buffers
        tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        for tensor in tensor_names:
            size = trt.volume(context.get_tensor_shape(tensor))
            dtype = trt.nptype(engine.get_tensor_dtype(tensor))
            
            if engine.get_tensor_mode(tensor) == trt.TensorIOMode.INPUT:
                context.set_input_shape(tensor, (2, 112, image_height, image_width))
                input_buffer = np.ascontiguousarray(input)
                input_memory = cuda.mem_alloc(input.nbytes)
                context.set_tensor_address(tensor, int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                context.set_tensor_address(tensor, int(output_memory))

        stream = cuda.Stream()
        
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        
        # Run inference
        start = time.time()
        context.execute_async_v3(stream_handle=stream.handle)
        end = time.time()
        print("fusion inference time:",(end - start))
        # Transfer prediction output from the GPU.
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        
        # Synchronize the stream
        stream.synchronize()

        return  np.reshape(output_buffer, (1, 112, image_height, image_width))
    
def decode_infer(engine, input, image_height, image_width):
    with engine.create_execution_context() as context:
        # Allocate host and device buffers
        tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        for tensor in tensor_names:
            size = trt.volume(context.get_tensor_shape(tensor))
            dtype = trt.nptype(engine.get_tensor_dtype(tensor))
            
            if engine.get_tensor_mode(tensor) == trt.TensorIOMode.INPUT:
                context.set_input_shape(tensor, (1, 112, image_height, image_width))
                input_buffer = np.ascontiguousarray(input)
                input_memory = cuda.mem_alloc(input.nbytes)
                context.set_tensor_address(tensor, int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                context.set_tensor_address(tensor, int(output_memory))

        stream = cuda.Stream()
        
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        
        # Run inference
        start = time.time()
        context.execute_async_v3(stream_handle=stream.handle)
        end = time.time()
        print("decode inference time:",(end - start))
        # Transfer prediction output from the GPU.
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        
        # Synchronize the stream
        stream.synchronize()

        return  np.reshape(output_buffer, (1,1, image_height, image_width))
    
def total_infer(engine, input_x, input_y, image_height, image_width):
    with engine.create_execution_context() as context:
        # Allocate host and device buffers
        tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        for tensor in tensor_names:
            size = trt.volume(context.get_tensor_shape(tensor))
            dtype = trt.nptype(engine.get_tensor_dtype(tensor))
            
            if engine.get_tensor_mode(tensor) == trt.TensorIOMode.INPUT and tensor == 'x':
                context.set_input_shape(tensor, (1, 1, 1152, 1440))
                x_buffer = np.ascontiguousarray(input_x)
                x_memory = cuda.mem_alloc(input_x.nbytes)
                context.set_tensor_address(tensor, int(x_memory))
            elif engine.get_tensor_mode(tensor) == trt.TensorIOMode.INPUT and tensor == 'y':
                context.set_input_shape(tensor, (1, 1, 1152, 1440))
                y_buffer = np.ascontiguousarray(input_y)
                y_memory = cuda.mem_alloc(input_y.nbytes)
                context.set_tensor_address(tensor, int(y_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                context.set_tensor_address(tensor, int(output_memory))

        stream = cuda.Stream()
        
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(x_memory, x_buffer, stream)
        cuda.memcpy_htod_async(y_memory, y_buffer, stream)
        
        # Run inference
        start = time.time()
        context.execute_async_v3(stream_handle=stream.handle)
        end = time.time()
        print("decode inference time:",(end - start))
        # Transfer prediction output from the GPU.
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        
        # Synchronize the stream
        stream.synchronize()

        return  np.reshape(output_buffer, (1,1, 1152, 1440))
    
def infer(encode_engine, fusion_session, decode_engine, ir_image, vis_image, image_height, image_width, output_file):
    X_data = []
    X_data.append(ir_image[0])
    X_data.append(vis_image[0])

    input_data = np.array(X_data)
    encoded_output = encode_infer(encode_engine, input_data, image_height, image_width)
    fused_output = fusion_infer(fusion_session, encoded_output, image_height, image_width)
    decoded_output = decode_infer(decode_engine, fused_output, image_height, image_width)
    
    utils.save_images(output_file, decoded_output[0])
    

def main():
    in_c = 1
    out_c = in_c
    image_height = 1152
    image_width = 1440
    parsers = get_parser()
    args = parsers.parse_args()
    test_path = args.test_path
    file_list = os.listdir(test_path+'ir/')
    output_path='./results/21Pairs/ResCbamFuse/'
    if os.path.exists(output_path) is False:
        os.makedirs(output_path)

    with torch.no_grad():

        #decoderEngine = load_engine(decoder_engine)
        #fusionEngine = load_engine(fusion_onnx)
        #encoderEngine = load_engine(encoder_engine)
        #EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        combinedengine = load_engine(combined_engine)

        for name in file_list:
            print(name)
            infrared_path = os.path.join(test_path, 'ir/' + name)
            visible_path = os.path.join(test_path,  'vi/' +name)

            ir_image = utils.get_test_images(infrared_path, height=image_height, width=image_width, mode='L')
            vis_image = utils.get_test_images(visible_path, height=image_height, width=image_width, mode='L')
            image = total_infer(combinedengine, ir_image, vis_image, image_height, image_width)
            utils.save_images(output_path + name, image[0])
            #infer(encoderEngine, fusionEngine, decoderEngine, ir_image, vis_image, image_height, image_width, f'E:\\Projects\\THERMAL_MONOCULAR\\ResCCFusion-Fork\\output\\{name}')



if __name__ == '__main__':
	main()