import cv2
import matplotlib.pyplot as plt
from utils import *

if __name__ == '__main__': 
    train_X, train_Y = readFile2('data/train.csv')

    train_X = train_X.astype(dtype = np.int8).reshape(-1, 28, 28, 1)
    # train_X = np.concatenate([train_X, train_X, train_X], axis = -1)
    # board = np.zeros((28 * 10, 28 * 10, 1), dtype = np.int8)

    # for i in range(10):
    #     row_index = i * 28
    #     for j in range(10):
    #         col_index = j * 28
    #         board[row_index: row_index + 28, col_index: col_index + 28] = train_X[i * 10 + j]

    # cv2.imshow('image', board)
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image', 600,600)
    # print(train_X[0].shape)
    # cv2.imshow('image', train_X[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    plt.matshow(train_X[0])
    plt.show()