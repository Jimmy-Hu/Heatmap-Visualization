import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm

import time
import os


# import data
def readFile(filename):
    f = open(filename, 'r')
    content = [row[:-1] for row in f.readlines()]
    f.close()

    title = content[0].split(',')
    if 'train' in filename:  # train data
        X = [data.split(',')[1:] for data in content[1:]]
        Y = [data[0] for data in content[1:]]
        # Y = []
        # for data in content[1:]:
        #     zeros = [0] * 10
        #     zeros[int(data[0])] = 1
        #     Y.append(zeros)

        return np.array(X, dtype = np.int32), np.array(Y, dtype = np.int32)

    elif 'test' in filename:  # test data
        return np.array([data.split(',') for data in content[1:]], dtype = np.int32)


# output coldmap with origin image height and width
def coldmap(image, cnn_out):
    heatmap = np.abs(np.mean(cnn_out, axis = 0))
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image = np.concatenate([image, image, image], axis = 0).transpose((1, 2, 0)) * 219
    # img = image
    img = heatmap * 0.3 + image
    img = img.astype(np.int8)

    return img


# output heatmap with origin image height and width
def heatmap2(image, cnn_out):
    heatmap = np.mean(cnn_out, axis = 0)
    heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))
    # img = image

    return heatmap


# output heatmap with origin image height and width
def heatmap(image, cnn_out):
    heatmap = np.mean(cnn_out, axis = 0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image = np.concatenate([image, image, image], axis = 0).transpose((1, 2, 0)) * 219
    # img = image
    img = heatmap * 0.3 + image
    img = img.astype(np.int8)

    return img


# display all heatmap image
def display(images, cnn_outs, row = 2, col = 3):
    img = np.zeros((28 * row, 28 * col, 3))
    for index, (image, cnn_out) in enumerate(zip(images, cnn_outs)):
        row_index = (index // 3) * 28
        col_index = (index % 3) * 28
        img[row_index: row_index + 28, col_index: col_index + 28] = heatmap(image, cnn_out)
        
    cv2.imshow('image', img)
    cv2.waitKey(100)


# display all heatmap image
def display2(fig, ax, images, cnn_outs, row = 2, col = 3, first = True):
    img = np.zeros((28 * row, 28 * col))
    for index, (image, cnn_out) in enumerate(zip(images, cnn_outs)):
        row_index = (index // 3) * 28
        col_index = (index % 3) * 28
        img[row_index: row_index + 28, col_index: col_index + 28] = heatmap2(image, cnn_out)
    
    row_range = list(range(28 * row))
    col_range = list(range(28 * col))
    col_range, row_range = np.meshgrid(col_range, row_range)

    print('row_range: {}, col_range: {}, img_shape: {}'.format(row_range.shape, col_range.shape, img.shape))

    ax.clear()
    surf = ax.plot_surface(col_range, row_range, img, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # fig.colorbar(surf, shrink = 0.5, aspect = 5)
    # fig.colorbar(surf, cax = cax)
    ax.view_init(90, 90)


def change_plot(zarray, plot):
   plot[0].remove()
   plot[0] = ax.plot_surface(x, y, zarray[:, :], cmap="afmhot_r")


if __name__ == '__main__':
    # load data example code
    train_X, train_Y = readFile('data/train.csv')
    test_X = readFile('data/test.csv')

    print('sample: %d, data dim: %d' %(len(train_X), len(train_X[0])))
    print('sample label: %d, data dim: %d' %(len(train_Y), len(train_Y[0])))
    print('sample: %d, data dim: %d' %(len(test_X), len(test_X[0])))
