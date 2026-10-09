'''
Author: xiafanli xiafanli@eccom.com.cn
Date: 2023-10-25 17:16:58
Description: 推理
'''
import sys
sys.path.append('core')
from core.utils.utils import InputPadder, coords_grid, upflow8
import time_helper
from core.utils import flow_viz
from core.raft import RAFT

import torch
import torch.nn.functional as F
import numpy as np
import cv2


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
    print('torch.cuda.amp.autocast is not available, please update PyTorch to version >= 1.6')


class OpticalFlowEstimation_pt():
    def __init__(self, args):
        self.args = args
        self.model = self.init_model()

    def init_model(self):
        self.model = torch.nn.DataParallel(RAFT(self.args))
        self.model.load_state_dict(torch.load(self.args.pt_model))
        self.model = self.model.module
        self.model.to(self.args.device)
        self.model.eval()
        return self.model

    def trans_img(self, img):
        '''
        将cv读取的图片转换为模型输入格式
        '''
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(self.args.device)

    def visualize_flow(self, img, flo):
        '''
        将光流可视化
        '''
        img = torch.from_numpy(img).float()
        flo = torch.from_numpy(flo).float().cpu().numpy()
        # flo = flo.permute(1, 2, 0).cpu().numpy()

        # map flow to rgb image
        flo = flow_viz.flow_to_image(flo)
        img_flo = np.concatenate([img, flo], axis=0)

        return img_flo[:, :, [2, 1, 0]]/255.0

    @time_helper.cost_time(f"<Pytorch Infer>")
    @torch.no_grad()
    def __call__(self, img1, img2):
        '''
        input : img1,img2:np.array 维度为[H, W, C]，其中C是通道数，H是高度，W是宽度。
        return :一个torch tensor，维度为[H, W，2]，其中2是x和y方向的光流。

        '''
        img1 = self.trans_img(img1)
        img2 = self.trans_img(img2)

        with torch.no_grad():
            padder = InputPadder(img1.shape)
            img1, img2 = padder.pad(img1, img2)
            flow_up = self.model(
                img1, img2, self.args.iters, test_mode=True)
            flow_up = padder.unpad(flow_up)
            flow = flow_up.detach().cpu()
            flow = flow[0].permute(1, 2, 0).numpy()

        return flow

if __name__ == "__main__":
    # 测试代码
    import cv2
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pt_model', default=r'F:\weights\RAFT_change\185000_raft_flow_replknet.pth', help="restore checkpoint")
    parser.add_argument(
        '--fnet_onnx', default=r'F:\weights\RAFT_change\5000_raft_change-sintel-r4-fnet.onnx', help="restore checkpoint")
    parser.add_argument(
        '--cnet_onnx', default=r'F:\weights\RAFT_change\5000_raft_change-sintel-r4-cnet.onnx', help="restore checkpoint")
    parser.add_argument('--updata_block_onnx',
                        default=r'F:\weights\RAFT_change\5000_raft_change-sintel-r4-update_block.onnx', help="restore checkpoint")
    parser.add_argument(
        '--onnx_model', default=r'F:\weights\RAFT_change\5000_raft_change-sintel-r4.onnx', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision',
                        action='store_true', help='use mixed precision')
    parser.add_argument('--device', default='cuda', help='device to use')
    parser.add_argument('--iters', type=int, default=6,
                        help='number of refinement iterations, default: 6')
    parser.add_argument('--image_size', type=int, nargs='+',
                        default=[1080, 1920], help='image size, default: [640, 640]')
    args = parser.parse_args()

    # pt-model
    pt_model = OpticalFlowEstimation_pt(args)

    # 输入
    video_path = r"G:\Training_Data_Hurunui_River\Hurunui Drone Video.mp4"
    # video_path = settings.speedconfig.video_path
    # 保存结果视频
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cv2.CAP_PROP_FPS
    out = cv2.VideoWriter('FLOW2_es4.mp4', fourcc, fps,(2160,3840))

    # 加载视频     
    cap = cv2.VideoCapture(video_path)
    pre_ret, pre_frame = cap.read()
    pre_frame = cv2.resize(pre_frame, (1080, 1920))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1080, 1920))
        pred = pt_model(pre_frame, frame)
        pre_frame = frame
        if pred is not None and pred.shape[0] > 0 and pred.shape[1] > 0:
                img_flo = pt_model.visualize_flow(pre_frame, pred)
                cv2.imshow('img_flo', img_flo)
                cv2.waitKey(0)
        else:
            print('Image is empty or has zero dimension.')

    out.release()

    # 可视化
    # print(flow_pt.shape)
    # img_flo = pt_model.visualize_flow(img1, flow_pt)
    # cv2.imshow('img_flo', img_flo)
    # cv2.waitKey(0)
