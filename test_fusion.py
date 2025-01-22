# test phase
import torch
from torch.autograd import Variable
from net_cbam import ResCCNet_cbam_fuse
# from utils import list_images,make_floor
import utils
from args_fusion import get_parser
import numpy as np
import os
import time

def load_model(path,  output_nc):
    model = ResCCNet_cbam_fuse(output_nc)
    model.load_state_dict(torch.load(path))
    model.eval()
    model.cuda()
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    return model

def _generate_fusion_image(model,img_ir,img_vis):
	return model.forward(img_ir,img_vis)

def run_demo(model, infrared_path, visible_path, output_path_root, index, network_type, mode, args):
	# prepare data
	
	ir_img = utils.get_test_images(infrared_path, height=1152, width=1440, mode=mode)
	vis_img = utils.get_test_images(visible_path, height=1152, width=1440, mode=mode)
	if args.cuda:
		ir_img = ir_img.cuda()
		vis_img = vis_img.cuda()
	ir_img = Variable(ir_img, requires_grad=False)
	vis_img = Variable(vis_img, requires_grad=False)

	# fuse images
	start = time.time()
	img_fusion = _generate_fusion_image(model, ir_img, vis_img)
	end = time.time()
	print("Total inference:",(end - start))


	# save images
	if args.cuda:
		img = img_fusion.to(torch.float).cpu().clamp(0, 255).data[0].numpy()
	else:
		img = img_fusion.clamp(0, 255).data[0].numpy()

	file_name1 =str(index)+'.png'
	output_path1 = output_path_root + '/'+ network_type + '/' + file_name1
	if os.path.exists(output_path_root + '/'+ network_type) is False:
		os.makedirs(output_path_root + '/'+ network_type)
	utils.save_images(output_path1, img)
	
def main():
	parsers = get_parser()
	args = parsers.parse_args()
	# run demo
	test_path = args.test_path

	file_list = os.listdir(test_path +'ir/')

	strategy_type =  'cbam'
	output_path = args.output_path
	if os.path.exists(output_path) is False:
		os.makedirs(output_path)

	in_c = 1
	out_c = in_c
	mode = 'L'
	model_path = args.model_path # ssim weight is 1
	ssim_name = model_path[-9:-6]
	network_type = 'ResDFuse_' + strategy_type + '_' + ssim_name

	with torch.no_grad():
		# print('SSIM weight ----- ' + args.ssim_path[0])
		model = load_model(model_path, out_c)
		totaltime=0
		for name in file_list:
			print(name)
			infrared_path = os.path.join(test_path, 'ir/' + name)
			visible_path = os.path.join(test_path,  'vi/' +name)
			start = time.time()
			run_demo(model, infrared_path, visible_path, output_path, name[:-4], network_type, mode, args)
			end = time.time()
			totaltime+=end-start
		print(strategy_type+':%.8f' % float(totaltime/len(file_list)))
	print(output_path)
	print('Done......')

if __name__ == '__main__':
	main()



