import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import json
from PIL import Image
from torch.nn import functional as F


# load the model or pre_model
def pre_model(model_id, pretrained):
    if model_id == 1:
        # pretrained 文件在.cache 里
        net = models.squeezenet1_1(pretrained=pretrained)
        finalconv_name = 'features' # the last conv layer output, before the classifier.
    elif model_id == 2:
        net = models.resnet18(pretrained=pretrained)
        finalconv_name = 'layer4'
    elif model_id == 3:
        net = models.densenet161(pretrained=pretrained)
        finalconv_name = 'features'
    print(net)
    return net, finalconv_name

def model(model_name):
    # you should konw the last conv layer name before.
    net = model_name
    print(net)
    return net
    
def preprocess(img_pil):
    """
    input_size: (Tuple:(int,int))    depend on your model's image input
    """
    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
    transforms.ToTensor(),
    ])
    return transform(img_pil)

# 计算grad-cam
def returnGradCAM(feature_map, grads):
    nc, h, w = feature_map.shape
    output_gradcam = []
    gradcam = np.zeros(feature_map.shape[1:], dtype=np.float32)	# feature_map.shape[1:]取第一维度后的尺寸，零初始化
    grads = grads.reshape([grads.shape[0],-1])					# 计算每个通道权重
    weights = np.mean(grads, axis=1)							# 权重均值
    for i, w in enumerate(weights):
        gradcam += w * feature_map[i, :, :]						# 梯度与对应权重相乘再累加
    gradcam = np.maximum(gradcam, 0)                            # 相当于ReLU操作
    gradcam = gradcam / gradcam.max()
    cam_img = np.uint8(255 * gradcam)
    cam_img = cv2.resize(cam_img, (80, 80))
    output_gradcam.append(cam_img)
    return output_gradcam

def GradCAM(model_name, pretrained, checkpoint_path, finalconv_name, device, img_path, oimg_path, label_path, out_name, out_o_name):
    # 存放梯度和特征图
    fmap_block = []
    grad_block = []
    # 定义获取梯度的函数，由于有多个全连接层，所以要依靠反向传播回溯，计算梯度值
    def backward_hook(module, grad_in, grad_out):
        grad_block.append(grad_out[0].detach())

    # 定义获取特征图的函数
    def farward_hook(module, input, output):
        fmap_block.append(output)

    if type(model_name) == int:
        net, finalconv_name = pre_model(model_name, pretrained)
        net._modules.get(finalconv_name).register_forward_hook(farward_hook)	# features最后一个卷积层
        # net.features[-1].expand3x3.register_backward_hook(backward_hook) # 这是针对squeezenet的一种写法
        net._modules.get(finalconv_name).register_backward_hook(backward_hook) # 一般获取最后一个卷积层可以这样写
    else:
        net = model(model_name)
        # check the last conv layer name
        for name, module in net.named_modules():
            print('modules:', name)
        # this finalconv_name need to write 
        finalconv_name = finalconv_name
        # hook 需要锁定的层名称在load前设置好，多gpu训练的模型load后名称会多'module'
        # [-3]是根据最后一个卷积层的位置得来
        net._modules.get(finalconv_name)[0].register_forward_hook(farward_hook)	# features最后一个卷积层
        net._modules.get(finalconv_name)[0].register_backward_hook(backward_hook) # 一般获取最后一个卷积层可以这样写

        if pretrained ==True:
            # net = nn.DataParallel(net)
            net.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
            for name, module in net.named_modules():
                print('modules:', name)  
    net.eval()
    # 在GPU训练后，用CPU载入模型参数会导致named_modules()返回值包含'module'，模型层名称变化，会使得hook方法报错，所以应该在载入参数前完成hook            
    # 这里获取softmax的权重，要清楚在net.parameters()中的哪个位置
    params = list(net.parameters())
    # print(params)
    # np.squeeze删除所有1维度
    weight_softmax = np.squeeze(params[-2].data.numpy())

    # 载入待测试的单张图像
    img_pil = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
    img_pil = np.expand_dims(img_pil,0).repeat(4,axis=0)
    # 图像三维数据解压为四维，即给一个batch_size=1, (1,3,X,X)
    # 训练数据可能是32x32的尺寸，但由于用卷积层分类，所以对其它尺寸也能识别，提前对输入图片进行放缩有助于增大最后一层特征图尺寸，可视化效果更好
    # 但原始尺寸太小，可视化后结果几乎不变，因为放大采用的插值算法
    # 因为我自己设计的model对输入尺寸没有特定要求，虽然是用32x32大小数据集训练的，但模型都包含avgpool层，能放缩最后一层卷积层的输出特征到统一尺寸，对分类层是fc还是conv没有影响。
    # 所以这里不需要对输入图片做resize
    img_tensor = preprocess(img_pil).unsqueeze(0).reshape(1,4,80,80)


    output = net(img_tensor)
    # label 的json文件
    json_path = label_path
    with open(json_path, 'r') as load_f:
        load_json = json.load(load_f)
    classes = {int(key): value for (key, value) in load_json.items()}
    # print(classes)

    h_x = F.softmax(output, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # output the prediction
    for i in range(0, 1):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # backward
    net.zero_grad()
    class_loss = output[0,idx]
    class_loss.backward(class_loss.clone().detach())

    # 获取grad
    grads_val = grad_block[0].cpu().data.numpy().squeeze(0) 
    # 当输入图片尺寸太小，或者模型深度太深时，会最终获得(1,C,1,1)的features_map，所以squeeze操作会把后面表示h*w的1也删去，从而报错
    fmap = fmap_block[0].cpu().data.numpy().squeeze(0)

    # 保存cam图片
    gradcam = returnGradCAM(fmap, grads_val)
    print('output' + out_name + '.jpg for the top1 prediction: %s'%classes[idx[0]])
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(gradcam[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite(out_name + '.jpg', result)

    oimg = cv2.imread(oimg_path)
    oheight, owidth, _ = oimg.shape
    oheatmap = cv2.applyColorMap(cv2.resize(gradcam[0],(owidth, oheight)), cv2.COLORMAP_JET)
    oresult = oheatmap * 0.3 + oimg * 0.5
    cv2.imwrite(out_o_name + '.jpg', oresult)
