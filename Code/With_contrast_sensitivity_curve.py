# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import cv2

# 图片地址
img_set_path = "dataset/wafers/From_ji_1/images/train/"

def get_my_freq(height, width, total_h, total_w, angle):
    freq_h = (3.14 * total_h) / (np.sin(angle/2) * 2 * height * 180)
    freq_w = (3.14 * total_w) / (np.sin(angle/2) * 2 * width * 180)
    #freq = np.sqrt(freq_h ** 2 + freq_w ** 2)
    freq = (freq_h + freq_w)/2
    freq = float(str(freq)[:4])
    #freq = int(freq)
    return freq

def get_my_sens(freq, angle, luminance):
    molecule = 5200 * np.exp(-0.0016 * (freq**2) * (1 + 100/luminance)**0.08)
    denominator = np.sqrt((1 + 144/angle**2 + 0.64*freq**2) * (63/luminance**0.83 + 1 / (1 - np.exp(-0.02*freq**2))))
    sens = molecule / denominator
    sens = float(str(sens)[:4])
    return sens

# 输入图像，输出超像素分割后的labels和masks
def superpixel_seg(image, reg_size, visible):
    superpixel = cv2.ximgproc.createSuperpixelSLIC(image, algorithm=cv2.ximgproc.SLIC, region_size=reg_size)
    superpixel.iterate()

    labels = superpixel.getLabels()
    #print("labels", labels)

    mask = superpixel.getLabelContourMask()

    if visible:
        ## 展示超像素分割图像
        result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result[mask == 255] = [255, 0, 0]
        plt.imshow(result)
        plt.show()
        ##
    return labels, mask

# 输入原图像、labels、masks，输出标注图像
def seg_with_sp(origin_img, labels, mask, gs_scale_list, size_scale_list, visible):
    #print("labels\n", labels)
    labels = np.array(labels)
    mask = np.array(mask)
    total_height = len(origin_img)
    total_width = len(origin_img[0])
    num_sp = np.max(labels) + 1
    #print(num_sp)

    ## 数组
    light_list = np.zeros(num_sp)
    size_list = np.zeros(num_sp)

    for i in range(num_sp):
        loc = np.where(labels == i)
        num_p = len(loc[0])
        size_list[i] = num_p
        for j in range(num_p):
            y = loc[0][j]
            x = loc[1][j]
            '''
            ##
            if type(light_list[i]) != int or type(origin_img[y, x]) != int:
                print("warn! ", light_list[i], origin_img[y, x])
            ##
            '''
            light_list[i] += origin_img[y][x]
        if num_p == 0:
            light_list[i] = 0
        else:
            light_list[i] = light_list[i] / float(num_p)

    aver_light = np.mean(light_list)
    #print("aver_light", aver_light)
    aver_size = np.mean(size_list)

    or_img = np.zeros((len(origin_img), len(origin_img[0])))
    k = 0
    for gs_scale in gs_scale_list:
        result = origin_img.copy()
        result = np.array(result)
        #plt.imshow(result, cmap="gray")
        #plt.xlabel("result0")
        #plt.show()
        size_scale = size_scale_list[k]
        for j in range(num_sp):
            light = light_list[j]
            size = size_list[j]
            ## 简单地手动设定“阈值”
            if light > aver_light*gs_scale or size < aver_size*(1-size_scale) or size > aver_size*(1+size_scale):
                result[labels == j] = 255
            else:
                result[labels == j] = 0
        if visible:
            plt.imshow(result, cmap="gray")
            plt.xlabel("result1")
            plt.show()

        connectivity = 4
        defect_num, defect_labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity, cv2.CV_32S)
        #print("a", a, "\n", "b", b, "\n", "stats", stats, "\n", "c", c)
        for vector in stats[1:]:
            x = vector[0]
            y = vector[1]
            width = vector[2]
            height = vector[3]
            # print(x, y, width, height)
            freq = get_my_freq(height, width, len(origin_img[0]), len(origin_img[1]), angle=30)
            sens = get_my_sens(freq, angle=30, luminance=np.mean(origin_img[y:y + height, x:x + width]))
            #sens = int(sens)

            ## sens阈值
            if sens < 120:
                result[y:y+height, x:x+width] = 0
            #else:
            #    print(sens)

        if visible:
            plt.imshow(result, cmap="gray")
            plt.xlabel("result2")
            plt.show()
        or_img = cv2.bitwise_or(or_img, result/255)
        if visible:
            plt.imshow(or_img, cmap="gray")
            plt.xlabel("or_img")
            plt.show()

        k += 1

    or_img = or_img * 255
    or_img = np.uint8(or_img)
    return or_img

# 输入二值图像，原图像，输出标注图像
def split_and_marking(binary_img, origin_img):
    #plt.imshow(binary_img, cmap="gray")
    #plt.savefig("run/cut_wafers/" + str(i) + ".jpg", dpi=300)
    '''
    min_gs = min(grayscale_list)
    max_gs= max(grayscale_list)
    grayscale_list = np.array(grayscale_list)
    grayscale_list = (grayscale_list - min_gs + 1) * (255 / (max_gs-min_gs+1))
    '''

    connectivity = 4
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity, cv2.CV_32S)

    edges = cv2.Canny(binary_img, 50, 100)
    edges = np.uint8(edges)
    show_img = cv2.cvtColor(origin_img, cv2.COLOR_GRAY2RGB)
    show_img[edges != 0] = [255, 0, 0]

    fig, ax = plt.subplots()
    ax.imshow(show_img)

    for vector in stats[1:]:
        x = vector[0]
        y = vector[1]
        width = vector[2]
        height = vector[3]
        #print(x, y, width, height)
        freq = get_my_freq(height, width, len(origin_img[0]), len(origin_img[1]), angle=30)
        sens = get_my_sens(freq, angle=30, luminance=np.mean(origin_img[y:y+height, x:x+width]))
        #if sens < 200:
        #    ax.text(x + width / 2, y + height + 20, "invisible", ha="center", color="red", fontsize=7)
        #else:
        #    ax.text(x + width / 2, y + height + 20, "visible", color="red", fontsize=7)
        #rect = plt.Rectangle((x, y), width, height, facecolor="none", edgecolor="red")
        #ax.add_patch(rect)
        #ax.text(x + width / 2, y + height + 20, "sens:" + str(sens), ha="center", color="red", fontsize=7)
        #ax.text(x + width / 2, y + height + 40, "freq:" + str(freq), ha="center", color="red", fontsize=7)
        '''
        ## 展示数据
        rect = plt.Rectangle((x, y), width, height, facecolor = "none", edgecolor="red")
        ax.add_patch(rect)
        #ax.text(x + width/2, y + height + 15, "dot", ha="center", color="red", fontsize=7)
        #ax.text(x + width / 2, y + height + 40, str(freq), ha="center", color="red", fontsize=7)
        #ax.text(x + width / 2, y + height + 20, "sens:"+str(sens), ha="center", color="red", fontsize=7)
        #ax.text(x + width / 2, y + height + 60, str(int(grayscale_list[k])), ha="center", color="red", fontsize=7)
        ##
        '''
    plt.show()
    #plt.savefig("run/cut_wafers/" + str(i) + "detect.jpg", dpi=300)
    plt.close()

    return labels, stats

if __name__ == '__main__':
    i = 1
    img = mpimg.imread(img_set_path + str(i) + ".jpg")

    '''
    ## 直接对原图做边缘检测
    edges = cv2.Canny(img, 20, 40)
    edges = np.uint8(edges)
    plt.imshow(edges, cmap="gray")
    plt.show()
    ##
    '''

    '''
    ## 对图像应用自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # clipLimit和tileGridSize根据实际情况进行调整
    enhanced_image = clahe.apply(img)
    plt.imshow(enhanced_image, cmap="gray")
    plt.show()
    ##

    ## 直方图均衡
    equalized = cv2.equalizeHist(img)
    plt.imshow(equalized, cmap="gray")
    plt.show()
    ##

    ## 均值迁移滤波法
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    result = cv2.pyrMeanShiftFiltering(rgb, 25, 10)
    plt.imshow(result)
    plt.show()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    result = np.array(result)
    max_1 = np.max(result)
    min_1 = np.min(result)
    result = (result - min_1) * (255 / (max_1-min_1))
    result = np.uint8(result)
    plt.imshow(result, cmap="gray")
    plt.xlabel("1")
    plt.show()
    ##
    '''

    final_or_img = np.zeros((len(img), len(img[0])))
    reg_size_list = [10, 20, 30]
    for reg_size in reg_size_list:
        labels, mask = superpixel_seg(img, reg_size=reg_size, visible=False)
        print("finish segmentation")
        gs_scale_list = [1.3, 1.2]
        size_scale_list = [0.3, 0.3]
        binary_img = seg_with_sp(img, labels, mask, gs_scale_list, size_scale_list, visible=False)
        final_or_img = cv2.bitwise_or(final_or_img, binary_img/255)
    final_or_img = np.uint8(final_or_img * 255)

    '''
    ## 显示合并二值图像、边缘检测和效果图
    plt.imshow(final_or_img, cmap="gray")
    plt.show()
    edges = cv2.Canny(final_or_img, 50, 100)
    edges = np.uint8(edges)
    plt.imshow(edges, cmap="gray")
    plt.show()
    show_img = cv2.bitwise_or(img, edges)
    plt.imshow(show_img, cmap="gray")
    plt.show()
    ##
    '''

    split_and_marking(final_or_img, img)