from utils import *
from model import cnn, Net

import random
import torch
import numpy as np
import cv2

if __name__ == '__main__': 
    iters = 10
    batch_size = 64

    model = cnn(1, 10)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

    # loading data(42000, 784)
    data_X, data_Y = readFile('data/train.csv')

    # seperate train data to train(32000) adn test(10000)
    test_X, test_Y = data_X[-10000:], data_Y[-10000:]
    train_X, train_Y = data_X[:-10000], data_Y[:-10000]

    # image preprocessing
    train_X = train_X.astype(np.float32).reshape(-1, 1, 28, 28) / 255.
    test_X = test_X.astype(np.float32).reshape(-1, 1, 28, 28) / 255.

    # cv2 API create window
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image', 600,600)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    for iter in range(iters):
        # random shuffle list sequence
        index_list = list(range(len(train_X)))
        random.shuffle(index_list)

        # training
        while len(index_list) != 0:
            model.train()
            # get batch size data index
            index = [index_list.pop(0) for _ in range(min(len(index_list), batch_size))]

            # transfer numpy data to torch tensor format
            X = torch.tensor(train_X[index], dtype = torch.float32)
            Y = torch.tensor(train_Y[index], dtype = torch.long)

            pred = model(X)
            loss = criterion(pred, Y)

            # reset model grad -> calculate loss and backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # view training weights response
            model.eval()
            X = torch.tensor(test_X[:6], dtype = torch.float32)
            pred, cnn_out = model(X, 'test')
            display2(fig, ax, test_X[:6], cnn_out.detach().numpy())

            # print('%d / %d - %5d: loss = %f' %(iters, iter, len(train_X) - len(index_list), loss.item()))

        # test
        preds = []
        index_list = list(range(len(test_X)))
        while len(index_list) != 0:
            index = [index_list.pop(0) for _ in range(min(len(index_list), batch_size))]

            X = torch.tensor(test_X[index], dtype = torch.float32)
            Y = torch.tensor(test_Y[index], dtype = torch.long)

            pred_Y = model(X)
            pred = torch.argmax(pred_Y, 1)
            for p in pred: preds.append(p)

        preds = np.array(preds)
        count = 0
        for i in range(len(test_X)):
            if preds[i] == test_Y[i]: count += 1

        print('accuracy: %.3f' %(count / len(test_X)))

    
    cv2.destroyAllWindows()

