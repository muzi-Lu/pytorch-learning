import cv2

# 读取两张图片
img1 = cv2.imread('1.jpg')
img2 = cv2.imread('2.jpg')

# 提取特征点和描述符
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 进行特征点匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# 匹配后选取最佳匹配点
matches = sorted(matches, key=lambda x: x.distance)
good_matches = matches[0:10]

# 绘制结果
result = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)
cv2.imshow('result', result)
cv2.waitKey()