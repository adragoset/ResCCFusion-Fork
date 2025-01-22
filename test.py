# test phase
import torch
from torch.autograd import Variable
from net import ResCCNet_atten_fuse
# from utils import list_images,make_floor
import utils
from args_fusion import get_parser
import numpy as np
import os
import time
import onnx
from onnxruntime import InferenceSession
from onnxconverter_common import float16
from onnxsim import simplify


def load_model(path,  output_nc):
	model = ResCCNet_atten_fuse(output_nc)
	model.load_state_dict(torch.load(path))
	model.eval()
	dummy_input = torch.randn(1, 1, 576, 720)
	# model.forward = model.encoder
	# torch.onnx.export(model,{ "input": encoder_dummy_input } ,"resccfusion_encoder.onnx", input_names=["input"])
	# simplified_encoder = onnx.load("resccfusion_encoder.onnx")
	# model_simp, check = simplify(simplified_encoder)
	# if(check == True):
	# 	onnx.save(model_simp, "resccfusion_encoder.onnx")

	# fusion_dummy_input = torch.randn(2, 112, 576, 720)
	# torch.onnx.export(model,{ "x": fusion_dummy_input, "y": fusion_dummy_input } ,"resccfusion.onnx", input_names=["x", "y"])
	# simplified_fusion = onnx.load("resccfusion_fusion.onnx")
	# model_simp, check = simplify(simplified_fusion)
	# if(check == True):
	# 	onnx.save(model_simp, "resccfusion.onnx")

	# decoder_dummy_input = torch.randn(1, 112, 576, 720)
	# model.forward = model.decoder
	# torch.onnx.export(model,{ "x": decoder_dummy_input } ,"resccfusion_decoder.onnx", input_names=["input"])
	# simplified_decoder = onnx.load("resccfusion_decoder.onnx")
	# model_simp, check = simplify(simplified_decoder)
	# if(check == True):
	# 	onnx.save(model_simp, "resccfusion_decoder.onnx")

	torch.onnx.export(model,{ "x": dummy_input, "y": dummy_input } ,"resccfusion.onnx", input_names=["x", "y"])
	simplified_encoder = onnx.load("resccfusion.onnx")
	model_simp, check = simplify(simplified_encoder)
	if(check == True):
		onnx.save(model_simp, "resccfusion.onnx")

	model = ResCCNet_atten_fuse(output_nc)
	model.load_state_dict(torch.load(path))
	model.eval()
	model.cuda()
	para = sum([np.prod(list(p.size())) for p in model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

	return model



def _generate_fusion_image(model,img_ir,img_vis,strategy_type,kernel_size = (8,1)):
	# encodestart = time.time()
	# en_ir = model.encoder(img_ir)
	# en_vis = model.encoder(img_vis)
	# encodeend = time.time()
	# print("Encode Time:",(encodeend - encodestart))
	# fusionstart = time.time()
	# feat = model.fusion(en_ir, en_vis, strategy_type,kernel_size)
	# fusionend = time.time()
	# print("Fusion Time:",(fusionend - fusionstart))
	# decodestart = time.time()
	# img_fusion = model.decoder(feat)
	# decodeend = time.time()
	# print("Decode Time:",(decodeend - decodestart))
	return model.forward(img_ir,img_vis)
	#return torch.from_numpy(model.run(None, {"x": img_ir, "y":img_vis})[0])


def run_demo(model, infrared_path, visible_path, output_path_root, index,  network_type, strategy_type, mode, args, kernel_size):
	# prepare data
	
	ir_img = utils.get_test_images(infrared_path, height=512, width=640, mode=mode)
	vis_img = utils.get_test_images(visible_path, height=512, width=640, mode=mode)
	if args.cuda:
		ir_img = ir_img.cuda()
		vis_img = vis_img.cuda()
	ir_img = Variable(ir_img, requires_grad=False)
	vis_img = Variable(vis_img, requires_grad=False)

	# fuse images
	start = time.time()
	img_fusion = _generate_fusion_image(model, ir_img, vis_img,strategy_type, kernel_size)
	end = time.time()
	print("Total inference:",(end - start))


	# save images
	if args.cuda:
		img = img_fusion.to(torch.float).cpu().clamp(0, 255).data[0].numpy()
	else:
		img = img_fusion.clamp(0, 255).data[0].numpy()

	file_name1 =str(index)+'_' +network_type +  '.png'
	output_path1 = output_path_root + '/'+ file_name1
	utils.save_images(output_path1, img)
	# print(output_path1)

def main():
	parsers = get_parser()
	args = parsers.parse_args()
	# run demo
	test_path = args.test_path

	file_list = os.listdir(test_path+'ir/')

	strategy_type =  'cc_atten'
	kernel_size = [[16,1], [8,1],[4,1],[2,1],[1,1]]
	kernel_size = kernel_size[1]
	output_path = args.output_path
	if os.path.exists(output_path) is False:
		os.makedirs(output_path)

	in_c = 1
	out_c = in_c
	mode = 'L'
	model_path = args.model_path # ssim weight is 1
	ssim_name = model_path[-9:-6]
	network_type = 'ResDFuse_'+strategy_type+ '_kernelsize_'+str(kernel_size[0])+'_'+ssim_name

	with torch.no_grad():
		# print('SSIM weight ----- ' + args.ssim_path[0])
		model = load_model(model_path, out_c)
		totaltime=0
		for name in file_list:
			print(name)
			infrared_path = os.path.join(test_path, 'ir/' + name)
			visible_path = os.path.join(test_path,  'vi/' +name)
			start = time.time()
			run_demo(model, infrared_path, visible_path, output_path, name[:-4], network_type,strategy_type,  mode, args, kernel_size)
			end = time.time()
			totaltime+=end-start
		print(strategy_type+':%.8f' % float(totaltime/len(file_list)))
	print(output_path)
	print('Done......')



if __name__ == '__main__':
	main()
