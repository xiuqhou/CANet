import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
import matplotlib.cm
# from torchinfo import summary
import copy
import cv2
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from mmdet.apis import (inference_detector,
                        init_detector, show_result_pyplot)

device = 'cpu'
config = r"work_dirs/genatt_cnxtblk_20221023_1255_faster_rcnn_r50_fpn_1x_WSJ_coco_0.877_noms/faster_rcnn_r50_fpn_1x_WSJ_coco.py"
ckpt = r"work_dirs/genatt_cnxtblk_20221023_1255_faster_rcnn_r50_fpn_1x_WSJ_coco_0.877_noms/latest.pth"
img = r"data/WSJ_2_coco/val2014/10.jpg"
score_thr = 0.5

# build the model from a config file and a checkpoint file
model = init_detector(config, ckpt, device=device)

# 定义列表用于存储中间层的输入或者输出
module_name = []
p_in = []
p_out = []

def forward_wrapper(name):
    def forward_hook(module, input, output):
        if type(input) == torch.Tensor:
            print(f"module name:{name},input shape:{input.shape},output shape:{output.shape}")
        else:
            print(f"module name:{name},input type:{type(input)},output type:{type(output)}")
        module_name.append(name)
        p_in.append(input)
        p_out.append(output)
    return forward_hook

# # 定义hook_fn，顾名思义就是把数值从
# def hook_fn(module, inputs, outputs):
#     print(f"module:{str(module)}")
#     module_name.append(module.__class__)
#     p_in.append(inputs)
#     p_out.append(outputs)

# model.bbox_head.retina_cls.register_forward_hook(hook_fn)
for name, module in model.named_modules():
    module.register_forward_hook(forward_wrapper(name))

# test a single image
result = inference_detector(model, img)
# show the results
# show_result_pyplot(model, img, result, score_thr=score_thr)

def show_feature_map(img_src, conv_features):
    '''可视化卷积层特征图输出
    img_src:源图像文件路径
    conv_feature:得到的卷积输出,[b, c, h, w]
    '''
    img = Image.open(img_src).convert('RGB')
    height, width = img.size
    conv_features = conv_features.cpu()

    heat = conv_features.squeeze(0)#降维操作,尺寸变为(2048,7,7)
    heatmap = torch.mean(heat,dim=0)#对各卷积层(2048)求平均值,尺寸变为(7,7)
    # heatmap = torch.max(heat,dim=1).values.squeeze()

    heatmap = heatmap.numpy()#转换为numpy数组
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)#minmax归一化处理
    heatmap = cv2.resize(heatmap,(img.size[0],img.size[1]))#变换heatmap图像尺寸,使之与原图匹配,方便后续可视化
    heatmap = np.uint8(255*heatmap)#像素值缩放至(0,255)之间,uint8类型,这也是前面需要做归一化的原因,否则像素值会溢出255(也就是8位颜色通道)
    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_HSV)#颜色变换
    plt.imshow(heatmap)
    plt.show()
    # heatmap = np.array(Image.fromarray(heatmap).convert('L'))
    superimg = heatmap*0.4+np.array(img)[:,:,::-1] #图像叠加，注意翻转通道，cv用的是bgr
    cv2.imwrite('./superimg.jpg',superimg)#保存结果
    # 可视化叠加至源图像的结果
    img_ = np.array(Image.open('./superimg.jpg').convert('RGB'))
    plt.imshow(img_)
    plt.show()

model_str = model.__str__()

if model_str.startswith('SSD'):
    for k in range(len(module_name)):
        for j in range(len(p_in[0][0])):
            print(p_in[k][0][j].shape)
            # print(p_out[k].shape)
            show_feature_map(img, p_in[k][0][j])
            # show_feature_map(img_file, torch.sigmoid(p_out[k]))
            print()

else:
    for k in range(len(module_name)):
        print(p_in[k][0].shape)
        print(p_out[k].shape)
        # show_feature_map(img_file, p_in[k][0])
        show_feature_map(img, torch.sigmoid(p_out[k]))
        print()