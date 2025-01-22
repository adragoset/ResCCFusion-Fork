import argparse

def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("--channels", type=int, default=1)
	parser.add_argument("--model_path", type=str, default="E:\\Projects\\THERMAL_MONOCULAR\\ResCCFusion-Fork\\models\\1e1\\Final_epoch_4_1e1cbam.model")
	parser.add_argument("--file_name", type=str, default="rescbamfusion")
	parser.add_argument("--base_onnx_save_dir", type=str, default="./models/onxx/")
	parser.add_argument("--base_trt_save_dir", type=str, default="./models/trt/")