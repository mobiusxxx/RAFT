# 网络部分使用onnx推理

import time
import torch.nn.functional as F
import sys
sys.path.append('core')
from core.utils.utils import coords_grid, upflow8, InputPadder
from core.corr import CorrBlock
import onnxruntime as ort
import torch
import numpy as np
from core.raft import RAFT
import cv2
import numpy as np
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


class OrtBase:
    def __init__(self, onnx_path, device):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device != 'cpu' else [
            'CPUExecutionProvider']
        self.session = ort.InferenceSession(
            path_or_bytes=onnx_path, providers=providers)
        self.input_name = [node.name for node in self.session.get_inputs()]
        self.output_name = [node.name for node in self.session.get_outputs()]
        print(f"Load model from <{onnx_path}> done.")
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    # @time_helper.cost_time(f"<Ort Infer>")
    def forward(self, inputs, *args, **kwargs):
        """
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param inputs:
        :return:
        """
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        if len(self.input_name) == 1:
            input_name = self.input_name[0]
            input_feed = {input_name: np.array(inputs.cpu())}
        else:
            assert len(self.input_name) == len(inputs)
            input_feed = {input_name: np.array(input_data.cpu()) for input_name,
                          input_data in zip(self.input_name, inputs)}
        predict = self.session.run(self.output_name, input_feed=input_feed)
        return predict


class OpticalFlowEstimation():
    def __init__(self, args):
        self.args = args
        self.hidden_dim = 128
        self.context_dim = 128
        self.corr_radius = 4
        self.fnet = OrtBase(
            onnx_path="5000_raft_change-sintel-r4-fnet.onnx", device='cuda')
        self.cnet = OrtBase(
            onnx_path="5000_raft_change-sintel-r4-cnet.onnx", device='cuda')
        self.update_block = OrtBase(
            onnx_path="5000_raft_change-sintel-r4-update_block.onnx", device='cuda')

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

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)
    
    @torch.no_grad()
    def __call__(self, image1, image2):
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        hdim = self.hidden_dim
        cdim = self.context_dim
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet.forward(
                inputs=[image1, image2])

        fmap1 = torch.from_numpy(fmap1).cuda()
        fmap2 = torch.from_numpy(fmap2).cuda()

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet.forward(inputs=image1)
            cnet = torch.from_numpy(cnet[0]).cuda()
            # net 为 GRU 的隐状态，inp 后续与其他特征结合作为 GRU 的一般输入
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        for itr in range(self.args.iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # 从相关性查找表中获取当前坐标的对应领域特征

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block.forward(inputs=(net, inp, corr, flow))  # 迭代更新模块
                up_mask = torch.from_numpy(up_mask).cuda()
                delta_flow = torch.from_numpy(delta_flow).cuda()
                net = torch.from_numpy(net).cuda()
            # print('update time:', time.time() - t6)
            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

        return flow_up
    
# @torch.no_grad()
# def test(model):
#     model.eval()
#     data1 = torch.randn(1, 3, 640, 640, device='cuda')
#     data2 = torch.randn(1, 3, 640, 640, device='cuda')
#     for i in range(10):
#         t1 = time.time()
#         with torch.no_grad():
#             flow3 = model(data1, data2)
#         print('time3:', time.time() - t1)


if __name__ == '__main__':
    # 测试onnx推理
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--image_size', type=int, nargs='+', default=[640, 640], help='image size for inference')
    parser.add_argument('--device', type=str, default='cuda:0', help='GPUs to use')
    parser.add_argument('--iters', type=int, default=6, help='number of refinement iterations')
    args = parser.parse_args()
    if args.mixed_precision:
        print('Using mixed precision.')
    onnx_seg = OpticalFlowEstimation(args)
    # input1 = cv2.imread(r'F:\datasets\water_datasets\create_data\training\img1\img1_1.png')
    # input2 = cv2.imread(r'F:\datasets\water_datasets\create_data\training\img2\img2_1.png')
    # input1 = torch.from_numpy(input1).permute(2, 0, 1)
    # input2 = torch.from_numpy(input2).permute(2, 0, 1)
    # data1 = input1[None].float().cuda()
    # data2 = input2[None].float().cuda()
    # data11 = torch.randint(0, 255, (1, 3, 640, 640), device='cuda').float()
    # data22 = torch.randint(0, 255, (1, 3, 640, 640), device='cuda').float()
    data111 = np.random.randint(0, 255, (1, 3, 640, 640)).astype(np.float32)
    data222 = np.random.randint(0, 255, (1, 3, 640, 640)).astype(np.float32)
    data1 = torch.from_numpy(data111).cuda()
    data2 = torch.from_numpy(data222).cuda()


    model = torch.nn.DataParallel(RAFT(args)).cuda()
    model.load_state_dict(torch.load(
    "F:/weights/RAFT_change/5000_raft_change-sintel-r4.pth"))
    model.module.eval()
    # with torch.no_grad():
    #     test(model.module)
    onnx = OrtBase("5000_raft_change-sintel-r4.onnx", device='cuda')
    for i in range(10):
        t1 = time.time()
        flow1 = onnx.forward(inputs=(data1,data2))[0][0]
        print('time1:', time.time() - t1)

    for i in range(10):
        t1 = time.time()
        flow2 = onnx_seg(data1, data2)[0].detach().cpu().numpy()
        print('time2:', time.time() - t1)
    
    for i in range(10):
        t1 = time.time()
        with torch.no_grad():
            padder = InputPadder(data1.shape)
            data1, data2 = padder.pad(data1, data2)
            flow3 = model(data1, data2)[0].detach().cpu().numpy()
        print('time3:', time.time() - t1)

    # print(np.sum(flow1-flow2.cpu().numpy()))
    print(np.sum(np.abs(flow2-flow3)))

