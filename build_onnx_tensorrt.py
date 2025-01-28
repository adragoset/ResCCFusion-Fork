import torch
import os
from torch.autograd import Variable
from net_cbam import ResCCNet_cbam_fuse
# from utils import list_images,make_floor
from args_converter import get_parser
import onnx
from onnxsim import simplify
import argparse
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def main():
    parsers = get_parser()
    args = parsers.parse_args()
    in_c = args.channels
    out_c = in_c
    model_path = args.model_path # ssim weight is 1
    model, dummy_input = load_model(out_c, in_c, model_path)
    file_name = args.file_name
    base_onnx_save_dir = args.base_onnx_save_dir
    base_trt_save_dir = args.base_trt_save_dir
    if os.path.exists(base_onnx_save_dir) is False:
        os.makedirs(base_onnx_save_dir)
    if os.path.exists(base_trt_save_dir) is False:
        os.makedirs(base_trt_save_dir)
    build_onx(model, dummy_input, base_onnx_save_dir, file_name)
    build_engine_from_onnx(base_trt_save_dir, base_onnx_save_dir, file_name)

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def load_model(input_nc, output_nc, path):
    dummy_input = torch.randn(1, input_nc, 1152, 1440)
    model = ResCCNet_cbam_fuse(input_nc, output_nc)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model,dummy_input

def build_onx(model, dummy_input, save_dir, filename):
    filepath = save_dir + filename + ".onnx"
    torch.onnx.export(model,{ "x": dummy_input, "y": dummy_input } ,filepath, input_names=["x", "y"])
    simplified_encoder = onnx.load(filepath)
    model_simp, check = simplify(simplified_encoder)
    if(check == True):
        onnx.save(model_simp, filepath)

def build_engine_from_onnx(base_trt_dir,base_onnx_dir, filename, encoder=False, decoder=False):
    engine = None
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
       
        path = base_onnx_dir + filename + ".onnx"
        # Parse model file
        assert os.path.exists(path), f'cannot find {path}'

        print(f'Loading ONNX file from path {path}...')
        with open(path, 'rb') as fr:
            if not parser.parse(fr.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                assert False

        print("Start to build Engine")
        plan = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(plan)
        

        plan = engine.serialize()
        savepth = base_trt_dir + filename + '.trt'
        with open(savepth, "wb") as fw:
            fw.write(plan)
            print('Finish conversion')

if __name__ == '__main__':
	main()