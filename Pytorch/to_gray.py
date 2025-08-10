import cv2

# 读取彩色图像（OpenCV默认读取为BGR格式）
img = cv2.imread("color_image.jpg")

# 转换为灰度图像
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 保存或显示
cv2.imwrite("gray_image.jpg", gray_img)
cv2.imshow("Gray Image", gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()