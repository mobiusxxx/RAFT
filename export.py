'''
Author: xiafanli xiafanli@eccom.com.cn
Date: 2023-10-23 16:06:53
Description: 
'''
import torch.nn as nn
import argparse
import torch
import sys
import onnxruntime

sys.path.append('core')
from core.raft import RAFT
import time
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true',
                    help='use mixed precision')
parser.add_argument('--image_size', type=int, nargs='+', default=[640, 640])
args = parser.parse_args()


model = torch.nn.DataParallel(RAFT(args)).cuda()
model.load_state_dict(torch.load(
    "F:/weights/RAFT_change/5000_raft_change-sintel-r4.pth"))
model.module.eval()

input1 = cv2.imread(r'F:\datasets\water_datasets\create_data\training\img1\img1_0.png')
input2 = cv2.imread(r'F:\datasets\water_datasets\create_data\training\img2\img2_0.png')
input1 = torch.from_numpy(input1).permute(2, 0, 1)
input2 = torch.from_numpy(input2).permute(2, 0, 1)
input1 = input1.unsqueeze(0).float().cuda()
input2 = input2.unsqueeze(0).float().cuda()
# input1 = 2 * (input1 / 255.0) - 1.0
# input2 = 2 * (input2/ 255.0) - 1.0

# input1 = input1.contiguous()
# input2 = input2.contiguous()
iter = 1 
#pt推理
t1=0
for i in range(iter):
    with torch.no_grad():
        time_temp = time.time()
        flow_up1 = model(input1, input2)
        t1+=time.time()-time_temp
print('time1:', t1/iter)


input_list = ['img1', 'img2']
output_list = ['output']
torch.onnx.export(model.module, (input1, input2), "5000_raft_change-sintel-r4.onnx", opset_version=16,verbose=False,
                  input_names=input_list, output_names=output_list,
                  dynamic_axes={'img1': {0: 'batch_size'}, 'img2': {
                      0: 'batch_size'}, 'output': {0: 'batch_size'}},
                  )
#onnx推理
onnx_model = onnxruntime.InferenceSession("5000_raft_change-sintel-r4.onnx")
input_name = onnx_model.get_inputs()[0].name
input_name1 = onnx_model.get_inputs()[1].name
t2 = 0
for i in range(iter):
    t3 = time.time()
    flow_up2 = onnx_model.run(['output'], {input_name: input1.cpu().numpy(),input_name1: input2.cpu().numpy()})
    t2 += time.time() - t3
print('time2:', t2/iter)
print(np.abs(flow_up1.cpu().numpy()[0]-flow_up2[0][0]).mean())   




# #转换torchscript
# traced_script_module = torch.jit.trace(model.module,(input1, input2))

# t1 = time.time()
# flow_up = traced_script_module(input1, input2)
# print('time:', time.time() - t1)

# traced_script_module.save("10000_raft_change-sintel-12-jit.pth")

#分部分导出onnx模型
input_list_fnet = ['img1', 'img2']
output_list_fnet = ['output1','output2']
input1 = 2 * (input1 / 255.0) - 1.0
input2 = 2 * (input2/ 255.0) - 1.0

input1 = input1.contiguous()
input2 = input2.contiguous()
torch.onnx.export(model.module.fnet, ([input1, input2]), "5000_raft_change-sintel-r4-fnet.onnx", do_constant_folding=True,opset_version=16,
                    input_names=input_list_fnet, output_names=output_list_fnet,
                    dynamic_axes={'img1': {0: 'batch_size'}, 'img2': {
                        0: 'batch_size'}, 'output1': {0: 'batch_size'},'output2':{0:'batch_size'}},
                    )
input_list_cnet = ['img1']
output_list_cnet = ['output']
torch.onnx.export(model.module.cnet, (input1), "5000_raft_change-sintel-r4-cnet.onnx", do_constant_folding=True,opset_version=16,
                    input_names=input_list_cnet, output_names=output_list_cnet,
                    dynamic_axes={'img1': {0: 'batch_size'}, 'output': {
                        0: 'batch_size'}},
                    )
net_shape = (128,80,80)
net = torch.randn(args.batch_size, *net_shape).cuda()
inp_shape = (128,80,80)
inp = torch.randn(args.batch_size, *inp_shape).cuda()
corr_shape = (324,80,80)
corr = torch.randn(args.batch_size, *corr_shape).cuda()
flow_shape = (2,80,80)
flow = torch.randn(args.batch_size, *flow_shape).cuda()

input_list_update_block = ['net', 'inp','corr','flow']
output_list_update_block = ['output1','output2','output3']
torch.onnx.export(model.module.update_block, (net , inp, corr, flow), "5000_raft_change-sintel-r4-update_block.onnx", do_constant_folding=True,opset_version=16,
                    input_names=input_list_update_block, output_names=output_list_update_block,
                    dynamic_axes={'net': {0: 'batch_size'}, 'inp': {
                        0: 'batch_size'}, 'corr': {0: 'batch_size'},'flow':{0:'batch_size'},'output1':{0:'batch_size'},'output2':{0:'batch_size'},'output3':{0:'batch_size'}},
                    )

