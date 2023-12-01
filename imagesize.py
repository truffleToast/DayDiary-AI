import cv2
mask=cv2.imread("./mask.png", cv2.IMREAD_COLOR)
img =cv2.imread("./img.png", cv2.IMREAD_COLOR)

print(mask.shape, img.shape)
