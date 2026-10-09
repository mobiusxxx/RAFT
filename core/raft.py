import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder,Flow_Attention
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8
from replknet import RepLKNet
import time
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        self.fnet = RepLKNet(large_kernel_sizes=[31,29], layers=[2,2], channels=[128,256],
                    drop_path_rate=0.3, small_kernel=5, use_checkpoint=True,
                    small_kernel_merged=False,num_classes=None,out_indices=[1])
        self.cnet = RepLKNet(large_kernel_sizes=[31,29], layers=[2,2], channels=[128,256],
                    drop_path_rate=0.3, small_kernel=5, use_checkpoint=True,
                    small_kernel_merged=False,num_classes=None,out_indices=[1])
        # self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
        # self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        # step 1：预处理
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # step 2：Feature Encoder 提取两图特征（权值共享）
        with autocast(enabled=self.args.mixed_precision):
            # fmap1, fmap2 = self.fnet([image1, image2])        
            fmap1  = self.fnet(image1)[0] 
            fmap2  = self.fnet(image2)[0]           
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        # step 3：初始化多尺度相关特征查询表
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius) 

        # step 4：Context Encode 上下文特征提取
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)[0]
            net, inp = torch.split(cnet, [hdim, cdim], dim=1) #net 为 GRU 的隐状态，inp 后续与其他特征结合作为 GRU 的一般输入
            net = torch.tanh(net)
            inp = torch.relu(inp)
          
        # step 5：更新光流
		# 初始化光流的坐标信息，coords0 为初始时刻的坐标，coords1 为当前迭代的坐标，此处两坐标数值相等
        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # 从相关性查找表中获取当前坐标的对应领域特征


            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)#迭代更新模块

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions
    
class RAFT_trans(nn.Module):
    def __init__(self, args):
        super(RAFT_trans, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        # self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.flow_attention = Flow_Attention(dim=256)        
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=6, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        # t1 = time.time()
        # step 1：预处理
        B, C, H, W = image1.shape
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim
        # print('preprocess time:', time.time() - t1)
        # t2 = time.time()
        # step 2：Feature Encoder 提取两图特征（权值共享）
        with autocast(enabled=self.args.mixed_precision):
            image1_patch = image1.view(-1,image1.shape[2]*image1.shape[3]*3//256,256)
            image2_patch = image2.view(-1,image2.shape[2]*image2.shape[3]*3//256,256)
            # fmap1, fmap2 = self.fnet([image1, image2])   
            fmap1, fmap2 = self.flow_attention([image1_patch, image2_patch])
            #下采样
            fmap1 = fmap1.view(B,-1,H,W)
            fmap2 = fmap2.view(B,-1,H,W)
            # fmap1_downsampled = F.interpolate(fmap1, scale_factor=0.125, mode='bilinear', align_corners=False)   
            # fmap2_downsampled = F.interpolate(fmap2, scale_factor=0.125, mode='bilinear', align_corners=False) 
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        # print('feature encoder time:', time.time() - t2)
        # t3 = time.time()
        # step 3：初始化多尺度相关特征查询表
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius) 
        # print('corr time:', time.time() - t3)
        # t4 = time.time()
        # step 4：Context Encode 上下文特征提取
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1) #net 为 GRU 的隐状态，inp 后续与其他特征结合作为 GRU 的一般输入
            net = torch.tanh(net)
            inp = torch.relu(inp)
        # print('context encoder time:', time.time() - t4)
        # t5 = time.time()
        # step 5：更新光流
		# 初始化光流的坐标信息，coords0 为初始时刻的坐标，coords1 为当前迭代的坐标，此处两坐标数值相等
        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # 从相关性查找表中获取当前坐标的对应领域特征


            flow = coords1 - coords0
            # t6 = time.time()
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)#迭代更新模块
            # print('update time:', time.time() - t6)
            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            # t7 = time.time()
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            # print('upsample time:', time.time() - t7)
            flow_predictions.append(flow_up)

        # print('total time:', time.time() - t1)
        #推理时，返回光流预测结果
        if test_mode:
            return coords1 - coords0, flow_up
        #训练时，返回每次迭代的光流预测结果
        
        return flow_predictions

#测试
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[440, 1024])
    parser.add_argument('--gpus', type=int, nargs='+', default=[1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()
    raft = RAFT(args).to('cuda:1')
    data1 = torch.randn(2, 3, 640, 640,device='cuda:1')
    data2 = torch.randn(2, 3, 640, 640,device='cuda:1')
    i = time.time()
    flow_low, flow_up = raft(data1, data2, iters=12, test_mode=True)
    print(time.time() - i)
    print(flow_low.shape)
