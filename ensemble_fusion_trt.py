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
from args_ensemble_fusion import get_parser



TRT_LOGGER = trt.Logger()

# Filenames of TensorRT plan file and input/output images.

def load_engine(path):
    print("Reading engine from file {}".format(path))
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
    
def total_infer(engine, input_x, input_y, channels, image_height, image_width):
    with engine.create_execution_context() as context:
        # Allocate host and device buffers
        tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        for tensor in tensor_names:
            size = trt.volume(context.get_tensor_shape(tensor))
            dtype = trt.nptype(engine.get_tensor_dtype(tensor))
            
            if engine.get_tensor_mode(tensor) == trt.TensorIOMode.INPUT and tensor == 'x':
                context.set_input_shape(tensor, (1, channels, image_height, image_width))
                x_buffer = np.ascontiguousarray(input_x)
                x_memory = cuda.mem_alloc(input_x.nbytes)
                context.set_tensor_address(tensor, int(x_memory))
            elif engine.get_tensor_mode(tensor) == trt.TensorIOMode.INPUT and tensor == 'y':
                context.set_input_shape(tensor, (1, channels, image_height, image_width))
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

        return  np.reshape(output_buffer, (1, channels, image_height, image_width))   

def main():
    image_height = 1152
    image_width = 1440
    parsers = get_parser()
    args = parsers.parse_args()
    mode = args.mode
    if mode == "L":
        in_c = 1
    else:
        in_c = 3
    test_path = args.test_path
    file_list = os.listdir(test_path+'ir/')
    output_path=args.output_path
    model_path=args.model_path
    if os.path.exists(output_path) is False:
        os.makedirs(output_path)

    with torch.no_grad():
        engine = load_engine(model_path)

        for name in file_list:
            print(name)
            infrared_path = os.path.join(test_path, 'ir/' + name)
            visible_path = os.path.join(test_path,  'vi/' +name)

            ir_image = utils.get_test_images(infrared_path, height=image_height, width=image_width, mode=mode)
            vis_image = utils.get_test_images(visible_path, height=image_height, width=image_width, mode=mode)
            image = total_infer(engine, ir_image, vis_image, in_c, image_height, image_width)
            utils.save_images(output_path + name, image[0])

if __name__ == '__main__':
	main()