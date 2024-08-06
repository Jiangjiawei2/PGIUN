# -- coding: utf-8 --

import csv
import os.path

import matplotlib.pyplot as plt


def visualize(csv_dir, csv_name):
    Loss = {}
    PSNR = {}
    SSIM = {}
    NRMSE = {}

    with open(os.path.join(csv_dir, csv_name)) as f:
        reader = csv.reader(f)
        header = next(reader)  # 使读取CSV文件中不包含表头
        for i in reader:
            if not i:
                continue
            else:
                Loss[int(i[0])] = float(i[1])
                PSNR[int(i[0])] = float(i[2])
                SSIM[int(i[0])] = float(i[3])
                NRMSE[int(i[0])] = float(i[4])
    plt.subplots_adjust(wspace=0.6, hspace=0.6)
    plt.subplot(221), plt.plot(Loss.keys(), Loss.values(), color='green'), plt.xlabel(header[0]), plt.ylabel(header[1])
    plt.subplot(222), plt.plot(PSNR.keys(), PSNR.values(), color='green'), plt.xlabel(header[0]), plt.ylabel(header[2])
    plt.subplot(223), plt.plot(SSIM.keys(), SSIM.values(), color='green'), plt.xlabel(header[0]), plt.ylabel(header[3])
    plt.subplot(224), plt.plot(NRMSE.keys(), NRMSE.values(), color='green'), plt.xlabel(header[0]), plt.ylabel(
        header[4])

    save_jpeg = os.path.join(csv_dir, 'metrics_epoch.jpeg')
    plt.savefig(save_jpeg)

# xiaoliang = {}
# # 读取csv文件，将文件中的数据绘制出来
# with open("C:\\Users\\Administrator\\Desktop\\xiaoliang.csv") as f:
#     reader = csv.reader(f)
#     header = next(reader)
#     # print(header)
#     for i in reader:
#         xiaoliang[int(i[0])] = int(i[1])  # 此处数据必须为int型
# plt.scatter(xiaoliang.keys(), xiaoliang.values())  # 散点图
# plt.plot(xiaoliang.keys(), xiaoliang.values())  # 连续图
# plt.title("《python入门销量》", fontproperties="SimHei")  # fontproperties解决中文不显示问题
# plt.xlabel(header[0], fontproperties="SimHei")
# plt.ylabel(header[1], fontproperties="SimHei")
# plt.show()
