import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import datasets
from utils import flow_viz
from utils import frame_utils

from raft import RAFT
from utils.utils import InputPadder, forward_interpolate

from tqdm import *
@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence, 'png')
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            flo_png = flow_viz.flow_to_image(flow)
            Image.fromarray(flo_png).save(output_file.replace('.flo', '.png'))
            # frame_utils.writeFlow(output_file, flow)

            sequence_prev = sequence

def visualize_flow(img, flo):
    '''
    将光流可视化
    '''
    img = torch.from_numpy(img).float()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    return flo[:, :, [2, 1, 0]]/255.0


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []
        t1 = 0
        for val_id in tqdm(range(len(val_dataset))):
            image1, image2, flow_gt,_  = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            t2 = time.time()
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
            # print("Elapsed time: %f seconds" % (time.time() - t2))
            t1+=time.time() - t2
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow- flow_gt.cpu())**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)
        print("Elapsed time(%s): %f seconds" % (dstype, t1))
        print("FPS(%s): %f" % (dstype,len(val_dataset) / (t1)))

    return results


@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}

@torch.no_grad()
def validate_self(model, args,iters=12):
    """ Peform validation using the create_data split """
    model.eval()
    epe_list = []

    val_dataset = datasets.realflow(split='validation')
    t1 = 0
    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        t3 = time.time()
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        # print("Preprocess time: %f seconds" % (time.time() - t3))

        t2 = time.time()
        flow_pr = model(image1, image2,iters=iters,test_mode=True)
        # print("Elapsed time: %f seconds" % (time.time() - t2))
        t1+=time.time() - t2

        t4 = time.time()
        flow = padder.unpad(flow_pr[0]).cpu()
        epe = torch.sum((flow - flow_gt.cpu())**2, dim=0).sqrt().cpu()
        epe_list.append(epe.view(-1).numpy())
        # print("EPE time: %f seconds" % (time.time() - t4))

    epe = np.mean(np.concatenate(epe_list))
    print("Validation realflow EPE: %f" % epe)
    print("Elapsed time: %f seconds" % (t1))
    print("FPS: %f" % (len(val_dataset) / (t1)))
    return {'realflow': epe}

@torch.no_grad()
def validate_LSPIV(model ,iters=12):
    """ Peform validation using the create_data split """
    model.eval()
    epe_list = []

    val_dataset = datasets.LSPIV(split='training')
    t1 = 0
    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        t3 = time.time()
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        # print("Preprocess time: %f seconds" % (time.time() - t3))


        t2 = time.time()
        _, flow_pr = model(image1, image2,iters=iters,test_mode=True)
        # print("Elapsed time: %f seconds" % (time.time() - t2))
        t1+=time.time() - t2

        t4 = time.time()
        flow = padder.unpad(flow_pr[0]).cpu()
        epe = torch.sum((flow - flow_gt.cpu())**2, dim=0).sqrt().cpu()
        epe_list.append(epe.view(-1).numpy())
        # print("EPE time: %f seconds" % (time.time() - t4))

    epe = np.mean(np.concatenate(epe_list))
    print("Validation LSPIV EPE: %f" % epe)
    print("Elapsed time: %f seconds" % (t1))
    print("FPS: %f" % (len(val_dataset) / (t1)))
    return {'LSPIV': epe}

# #预处理函数，改变图片tensor大小
# def preprocess(image,args):
#     """
#     input: image tensor of shape (3, H, W)
#     output: image tensor of shape (1, 3, H, W)
#     description: resize image 
#     """
#     image = image[None].cuda()
#     image = F.interpolate(image, size=(args.image_size[0],args.image_size[1]), mode='bilinear', align_corners=True)
#     return image
# #后处理函数，将图片返回原大小
# def postprocess(image,H,W):
#     """
#     input: image tensor of shape (1, 3, H, W)
#     output: image tensor of shape (3, H, W)
#     description: resize image 
#     """
#     image = F.interpolate(image, size=(H,W), mode='bilinear', align_corners=True)
#     image = image[0].cpu()
#     return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=r"F:\weights\RAFT_change\995000_Flowraft-sintel-batch6.pth",help="restore checkpoint")
    parser.add_argument('--dataset', default='LSPIV',help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--image_size', default=[384, 512], nargs='+', type=int, help='input image size')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    # torch.save(model.module, 'F:/weights/RAFT_change/raft_flow-sintel.pth')
    # model = torch.load(args.model)
    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    # data1 = torch.randn(1, 3, 640, 640, device='cuda')
    # data2 = torch.randn(1, 3, 640, 640, device='cuda')
    # with torch.no_grad():
    #     model.module.eval()
    #     for i in range(10):
    #         t10 = time.time()
    #         flow3 = model.module(data1, data2)
    #         print('time3:', time.time() - t10)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            validate_sintel(model.module)

        elif args.dataset == 'kitti':
            validate_kitti(model.module)
            
        elif args.dataset == 'self':
            validate_self(model.module)

        elif args.dataset == 'LSPIV':
            validate_LSPIV(model.module)


