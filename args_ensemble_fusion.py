import argparse

def get_parser():
	parser = argparse.ArgumentParser()
	#parser.add_argument("--mode", type=str, default="L")
	parser.add_argument("--mode", type=str, default="RGB")
	parser.add_argument("--model_path", type=str, default="E:\\Projects\\THERMAL_MONOCULAR\\ResCCFusion-Fork\\models\\trt\\rescbamfusionRGB.trt")
	parser.add_argument("--test_path", type = str, default="E:\\Projects\\THERMAL_MONOCULAR\\ResCCFusion-Fork\\testimages\\21pairs\\")
	parser.add_argument("--output_path", type = str, default=".\\results\\21Pairs\\ResCbamFuseRGB\\")
	return parser
	