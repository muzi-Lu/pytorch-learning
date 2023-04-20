import cv2
import numpy as np

from UNet import Unet, resnet34_unet
import torch
import torchvision.transforms as transforms
import numpy
import matplotlib.pyplot as plt


def Unet_feature_map():
    img = cv2.imread("frame-000000.color.png", flags=1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure()
    # plt.imshow(img)
    plt.imshow(imgRGB)
    plt.show()

    imgRGB = cv2.resize(imgRGB, (224, 224))
    imgRGB = (imgRGB / 255.0)

    # 将图片转换为numpy
    # img_arr = np.array(imgRGB)
    # print(img_arr)
    # 将numpy数组转换为tensor
    # img_tensor = torch.from_numpy(img_arr)
    # print(img_tensor)
    # img_tensor = img_tensor.unsquee

    transform = transforms.ToTensor()
    img_Tensor = transform(imgRGB)  # 形状和上面还不一样
    # img_Tensor = img_Tensor
    img_Tensor = img_Tensor.float()
    img_Tensor = img_Tensor.unsqueeze(0)
    # print(img_Tensor.shape)
    # Example:
    # x = torch.randn(3, 4)
    # x = x.unsqueeze(0)

    model = Unet(3, 3)
    # print(model)

    feature = model(img_Tensor)
    # print(feature.shape) # -> [1, 3, 224, 224]
    feature = feature.squeeze(0)
    # print(feature.shape) # -> [3, 224, 224]
    # feature.numpy()
    feature = feature.permute(2, 1, 0)
    feature_numpy = feature.detach().numpy()
    # print(feature_numpy.shape)
    # np.transpose(feature_numpy, (2, 1, 0)) # 这个等价permute但为什么不管用呢
    print(feature_numpy.shape)

    plt.figure()
    # plt.imshow(img)
    plt.imshow(feature_numpy)
    plt.show()


if __name__ == "__main__":
    Unet_feature_map()
