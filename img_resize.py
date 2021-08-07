# -*- coding: utf-8 -*-
# @Time    : 2021/7/3 16:30
# @Author  : He Ruizhi
# @File    : img_resize.py
# @Software: PyCharm

import cv2
img = cv2.imread(r'C:\Users\herz\Desktop\10.png')
img = cv2.resize(img, (474, 284))
# cv2.imshow('', img)
# cv2.waitKey()
cv2.imwrite(r'C:\Users\herz\Desktop\10.png', img)
