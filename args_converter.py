import argparse

def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("--channels", type=int, default=3)
	parser.add_argument("--model_path", type=str, default="E:\\Projects\\THERMAL_MONOCULAR\\ResCCFusion-Fork\\models\\1e1\\Final_epoch_4_1e1_cbam_RGB.model")
	parser.add_argument("--file_name", type=str, default="rescbamfusionRGB")
	parser.add_argument("--base_onnx_save_dir", type=str, default="./models/onnx/")
	parser.add_argument("--base_trt_save_dir", type=str, default="./models/trt/")
	return parser