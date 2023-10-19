# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import cv2

## 引用检测函数
import sys
#print(sys.path)
from With_contrast_sensitivity_curve import superpixel_seg, seg_with_sp, split_and_marking
##

## 图片地址
img_set_path_1 = "D:/learning things/summer exercitation/dataset/wafers/different features/" \
               "same_categories_szie diff_gray/"
img_set_path_2 = "D:/learning things/summer exercitation/dataset/wafers/different features/" \
               "same_categories_gray diff_size/"
img_set_path_3 = "D:/learning things/summer exercitation/dataset/wafers/different features/near_invisible/"

## 输入图像，输出检测结果
def run(img, visible):
    final_or_img = np.zeros((len(img), len(img[0])))
    reg_size_list = [20, 30]
    for reg_size in reg_size_list:
        labels, mask = superpixel_seg(img, reg_size=reg_size, visible=visible)
        print("finish segmentation")
        gs_scale_list = [1.5, 1.2]
        size_scale_list = [0.5, 0.4]
        binary_img = seg_with_sp(img, labels, mask, gs_scale_list, size_scale_list, visible=visible)
        final_or_img = cv2.bitwise_or(final_or_img, binary_img / 255)
    final_or_img = np.uint8(final_or_img * 255)
    split_and_marking(final_or_img, img)

if __name__ == '__main__':
    i = 1
    image = mpimg.imread(img_set_path_2 + str(i) + ".jpg")
    img = image.copy()
    img = np.array(img)
    #plt.imshow(img)
    #plt.show()

    ## 设置背景色
    height = len(img)
    width = len(img[0])
    for i in range(height):
        for j in range(width):
            pixel = img[i, j]
            #if 180<pixel[0]<185 and (pixel[0] != pixel[1] or pixel[1] != pixel[2] or pixel[2] != pixel[0]):
            if pixel[0] < 80:
                img[i, j] = [105, 105, 105]
    ##

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #plt.imshow(gray_img, cmap="gray")
    #plt.show()

    ## run
    run(gray_img, visible=False)