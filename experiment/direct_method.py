import numpy as np
import cv2

# 读取两幅图像
img1 = cv2.imread('img1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('img2.png', cv2.IMREAD_GRAYSCALE)

# 构建金字塔图像
pyramid1 = cv2.buildOpticalFlowPyramid(img1, 3)
pyramid2 = cv2.buildOpticalFlowPyramid(img2, 3)

# 构建图像梯度金字塔
dx1 = cv2.Sobel(pyramid1[0], cv2.CV_32F, 1, 0)
dy1 = cv2.Sobel(pyramid1[0], cv2.CV_32F, 0, 1)
grad1 = np.stack((dx1, dy1), axis=-1)
grad1 = cv2.GaussianBlur(grad1, (0, 0), 1)

dx2 = cv2.Sobel(pyramid2[0], cv2.CV_32F, 1, 0)
dy2 = cv2.Sobel(pyramid2[0], cv2.CV_32F, 0, 1)
grad2 = np.stack((dx2, dy2), axis=-1)
grad2 = cv2.GaussianBlur(grad2, (0, 0), 1)

# 构建图像金字塔
flow = np.zeros((pyramid1[0].shape[0], pyramid1[0].shape[1], 2), dtype=np.float32)
for i in range(2, -1, -1):
    # 对金字塔的当前层进行光流估计
    flow = cv2.pyrUp(flow, dstsize=pyramid1[i].shape[:2])
    flow = cv2.calcOpticalFlowFarneback(pyramid1[i], pyramid2[i], flow, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 对金字塔的当前层进行光流修正
    dx = cv2.resize(flow[..., 0], pyramid1[i - 1].shape[::-1], interpolation=cv2.INTER_LINEAR)
    dy = cv2.resize(flow[..., 1], pyramid1[i - 1].shape[::-1], interpolation=cv2.INTER_LINEAR)
    grad = np.stack((cv2.Sobel(pyramid1[i - 1], cv2.CV_32F, 1, 0),
                     cv2.Sobel(pyramid1[i - 1], cv2.CV_32F, 0, 1)), axis=-1)
    grad = cv2.GaussianBlur(grad, (0, 0), 1)
    A = np.stack((grad[..., 0] * dx, grad[..., 0] * dy, grad[..., 1] * dx, grad[..., 1] * dy), axis=-1)
    b = -grad[..., 0] * grad1[..., 0] - grad[..., 1] * grad1[..., 1]
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    flow[..., 0] += x[..., 0]
    flow[..., 1] += x[..., 1]

# 绘制光流效果
h, w = img1.shape[:2]
y, x = np.mgrid[0:h:10, 0:w:10].reshape(2, -1)
points = np.vstack((x, y)).T.astype(np.float32)
flow = flow[::10, ::10]
points2 = points + flow.reshape(-1, 2)
mask = np.ones_like(img1)
img3 = cv2.remap(img1, points2, None, cv2.INTER_LINEAR, None, cv2.BORDER_TRANSPARENT)
mask = cv2.remap(mask, points2, None, cv2.INTER_LINEAR, None, cv2.BORDER_CONSTANT)
result = cv2.addWeighted(img3, 0.5, img2, 0.5, 0, dtype=cv2.CV_8U)
cv2.imshow('result', result)
cv2.waitKey()