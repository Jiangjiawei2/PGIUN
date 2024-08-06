# -- coding: utf-8 --
import matplotlib.pyplot as plt


def save_image(path, data):
    plt.imsave(path, data, cmap='gray')
