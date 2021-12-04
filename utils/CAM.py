from PIL import Image
from torchvision import models, transforms
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
import torch
import json
import cv2


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
    transforms.ToTensor()
    ])
    return transform(img_pil)


"""
def register_forward_hook(self, hook):

       handle = hooks.RemovableHandle(self._forward_hooks)
       self._forward_hooks[handle.id] = hook
       return handle
这个方法的作用是在此module上注册一个hook，函数中第一句就没必要在意了，主要看第二句，是把注册的hook保存在_forward_hooks字典里。
hook 只能注册到 Module 上，即，仅仅是简单的 op 包装的 Module，而不是我们继承 Module时写的那个类，我们继承 Module写的类叫做 Container

当我们执行model(x)的时候，底层干了以下几件事：

1.调用 forward 方法计算结果
2.判断有没有注册 forward_hook，有的话，就将 forward 的输入及结果作为hook的实参。然后让hook自己干一些不可告人的事情。
"""

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cam_img)
    return output_cam


def CAM(model_name, pretrained, checkpoint_path, device, img_path, oimg_path, label_path, out_name, out_o_name):
    if type(model_name) == int:
        net, finalconv_name = pre_model(model_name, pretrained)
        features_blobs = []
        def hook_feature(module, input, output):
            features_blobs.append(output.data.cpu().numpy())
        net._modules.get(finalconv_name).register_forward_hook(hook_feature)
    else:
        net = model(model_name)
        # check the last conv layer name
        for name, module in net.named_modules():
            print('modules:', name)
        # this finalconv_name need to write 
        finalconv_name = 'conv3'
        # hook 需要锁定的层名称在load前设置好，load后会多'module'
        features_blobs = []
        def hook_feature(module, input, output):
            features_blobs.append(output.data.cpu().numpy())
        net._modules.get(finalconv_name)[0].register_forward_hook(hook_feature)
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

    # 载入待测试的单张图像并扩充置4个维度，满足模型输入尺寸
    img_pil = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
    img_pil = np.expand_dims(img_pil,0).repeat(4,axis=0)
    # 图像三维数据解压为四维，即给一个batch_size=1, (1,3,X,X)
    # 训练数据可能是32x32的尺寸，但由于用卷积层分类，所以对其它尺寸也能识别，提前对输入图片进行放缩有助于增大最后一层特征图尺寸，可视化效果更好
    # 但原始尺寸太小，可视化后结果几乎不变，因为放大采用的插值算法
    # 因为我自己设计的model对输入尺寸没有特定要求，虽然是用32x32大小数据集训练的，但模型都包含avgpool层，能放缩最后一层卷积层的输出特征到统一尺寸，对分类层是fc还是conv没有影响。
    # 所以这里不需要对输入图片做resize
    # img_tensor = preprocess(img_pil,256).unsqueeze(0)
    img_tensor = preprocess(img_pil).unsqueeze(0).reshape(1,4,80,80)


    logit = net(img_tensor)
    # label 的json文件
    json_path = label_path
    with open(json_path, 'r') as load_f:
        load_json = json.load(load_f)
    classes = {int(key): value for (key, value) in load_json.items()}
    # print(classes)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # output the prediction
    for i in range(0, 1):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))



    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    # render the CAM and output
    print('output' + out_name + '.jpg for the top1 prediction: %s'%classes[idx[0]])
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.2
    # result = heatmap * 0.3
    cv2.imwrite(out_name + '.jpg', result)

    oimg = cv2.imread(oimg_path)
    oheight, owidth, _ = oimg.shape
    oheatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(owidth, oheight)), cv2.COLORMAP_JET)
    oresult = oheatmap * 0.3 + oimg * 0.5
    cv2.imwrite(out_o_name + '.jpg', oresult)
