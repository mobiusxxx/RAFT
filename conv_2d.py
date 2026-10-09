'''
Author: xiafanli xiafanli@eccom.com.cn
Date: 2023-11-08 14:49:06
Description: 
'''

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import time_helper 
kernel_size = 5

@time_helper.cost_time("cal_field_from_store")
def cal_field_from_store(store_field):
    store_VA_field = np.mean(store_field, axis=-2)  # 时序 稳定值

    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)

    res_VA_Matrix = cv2.filter2D(store_VA_field, -1, kernel)  # 当前为空值像素，其搜索区域内数值的和
    nonzero_mask = store_VA_field != 0  # 统计store_VA_field一个kernel的非零值
    nonzero_number = cv2.filter2D(nonzero_mask.astype(np.float32), -1, kernel)  # 当前为空值像素，其搜索区域内非0的数量
    nonzero_number[nonzero_number == 0] = 1  # 避免除以0.当前为空值像素，其搜索区域内非0的数量也为0
    res_VA_Matrix = res_VA_Matrix / nonzero_number  # 空间上的平均值，搜索区域内的总和 / 搜索区域内非0的数量
    res_VA_Matrix[nonzero_mask] = store_VA_field[nonzero_mask]  # 当前为非空的像素，不作替换

    return res_VA_Matrix

class cal_field(nn.Module):
    def __init__(self, kernel_size=5):
        super(cal_field,self).__init__()
        self.kernel_size = kernel_size
        # self.conv = nn.Conv2d(2, 2, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.conv = nn.Conv2d(1, 1, kernel_size, padding=(kernel_size - 1)//2, bias=False,padding_mode='reflect')
        self.conv.weight.data.fill_(1.0)
    def conv2(self,input):
        c,h,w = input.shape
        x = torch.ones(c,h,w).cuda()
        for i in range(2):
            x[i] = self.conv(input[i].unsqueeze(0)).squeeze(0)
        return x
    @time_helper.cost_time("cal_field")
    def forward(self, store_field):
        store_VA_field = torch.mean(store_field, dim=-2)
        store_VA_field = store_VA_field.permute(2,0,1)
        res_VA_Matrix = self.conv2(store_VA_field)
        nonzero_mask = store_VA_field != 0
        nonzero_number = self.conv2(nonzero_mask.float())
        nonzero_number[nonzero_number == 0] = 1
        res_VA_Matrix = res_VA_Matrix / nonzero_number
        res_VA_Matrix[nonzero_mask] = store_VA_field[nonzero_mask]
        res_VA_Matrix = res_VA_Matrix.permute(1,2,0)

        return res_VA_Matrix
    
    def conv2_numpy(self,input):
        c,h,w = input.shape
        input = torch.from_numpy(input).float().cuda()
        input = input.permute(2,0,1)
        x = torch.zeros(c,h,w)
        x = x.permute(2,0,1)
        for i in range(2):
            x[i] = self.conv(input[i].unsqueeze(0)).squeeze(0)
        x = x.permute(1,2,0)
        return x.cpu().numpy()
    @time_helper.cost_time("cal_field_numpy")
    def con_forward(self,store_field):
        store_VA_field = np.mean(store_field, axis=-2)  # 时序 稳定值

        res_VA_Matrix = self.conv2_numpy(store_VA_field)  # 当前为空值像素，其搜索区域内数值的和
        nonzero_mask = store_VA_field != 0  # 统计store_VA_field一个kernel的非零值
        nonzero_number = self.conv2_numpy(nonzero_mask.astype(np.float32))  # 当前为空值像素，其搜索区域内非0的数量
        nonzero_number[nonzero_number == 0] = 1  # 避免除以0.当前为空值像素，其搜索区域内非0的数量也为0
        res_VA_Matrix = res_VA_Matrix / nonzero_number  # 空间上的平均值，搜索区域内的总和 / 搜索区域内非0的数量
        res_VA_Matrix[nonzero_mask] = store_VA_field[nonzero_mask]  # 当前为非空的像素，不作替换

        return res_VA_Matrix        
    
if __name__ == '__main__':
    #测试cal_field与cal_field_from_store的结果是否一致
    # store_field = np.random.randn(2160, 3840 , 10 , 2).astype(np.float32)
    #store_field中2通道中随机将100个位置的值置为0
    # store_field[np.random.randint(0, 2160, 1000), np.random.randint(0, 3840, 1000), :, 0] = 0
    store_field = np.load("filename.npy")
    store_field = torch.from_numpy(store_field).float().cuda()
    cal_field = cal_field().cuda()
    for i in range(10):
        with torch.no_grad():
            res_VA_Matrix = cal_field(store_field)
            # res_VA_Matrix1  = cal_field.con_forward(store_field.cpu().numpy())

    res_VA_Matrix_from_store = cal_field_from_store(store_field.cpu().numpy())
    res_VA_Matrix = res_VA_Matrix.cpu().numpy()
    print(np.sum(np.abs(res_VA_Matrix - res_VA_Matrix_from_store)))
    print(np.abs(res_VA_Matrix - res_VA_Matrix_from_store).mean())

    # #测试onnx导出
    # # input_list = ['input']
    # # output_list = ['output']
    # # torch.onnx.export(cal_field, store_field, "field_2d.onnx", do_constant_folding=True,opset_version=16,
    # #                 input_names=input_list, output_names=output_list,
    # #                 )

    kernel_size = 5
    conv = nn.Conv2d(1, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False,padding_mode='reflect')
    conv.weight.data.fill_(1.0)
    a = torch.ones(2, 2160, 3840)
    #对a分维度进行卷积
    convresult = torch.ones(2, 2160, 3840)
    for i in range(2):
        convresult[i] = conv(a[i].unsqueeze(0).unsqueeze(0))
    
    a_trans = a.permute(1, 2, 0)
    a_numpy = a_trans.numpy()
    convresult = convresult.permute(1,2,0)
    convresult_numpy = convresult.detach().numpy() 

    fliter2dresult = cv2.filter2D(a_numpy, -1, np.ones((kernel_size, kernel_size), dtype=np.float32))

    print(np.abs(convresult.detach().numpy() - fliter2dresult).mean())
