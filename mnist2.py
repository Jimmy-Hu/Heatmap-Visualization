from utils import *
from model import cnn, Net

import random
import torch
import numpy as np
import cv2

def extract(g):
    global x1g
    x1g = g

if __name__ == '__main__': 
    iters = 10
    batch_size = 64

    model = cnn(1, 10)
    conv2_param = list(model.parameters())[2]
    conv2_param.register_hook(extract)
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

    data_X, data_Y = readFile('data/train.csv')

    test_X, test_Y = data_X[-10000:], data_Y[-10000:]
    train_X, train_Y = data_X[:-10000], data_Y[:-10000]

    train_X = train_X.astype(np.float32).reshape(-1, 1, 28, 28) / 255.
    test_X = test_X.astype(np.float32).reshape(-1, 1, 28, 28) / 255.

    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image', 600,600)

    for iter in range(iters):
        index_list = list(range(len(train_X)))
        random.shuffle(index_list)
        conv2_grad = []
        # training
        while len(index_list) != 0:
            model.train()
            index = [index_list.pop(0) for _ in range(min(len(index_list), batch_size))]

            X = torch.tensor(train_X[index], dtype = torch.float32)
            Y = torch.tensor(train_Y[index], dtype = torch.long)

            pred = model(X)
            loss = criterion(pred, Y)

            # pred_list = pred.detach().tolist()
            # for i in range(len(pred_list)):
            #     print('label: {} - {}'.format(train_Y[index[i]], ' '.join(['%.3f' %p for p in pred_list[i]])))
                    
            # print(loss)
            # print('#########################3')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # view training weights response
            model.eval()
            # X = torch.tensor(test_X[:6], dtype = torch.float32)
            # pred, cnn_out = model(X, 'test')
            # display(test_X[:6], cnn_out.detach().numpy())

            pred, cnn_out = model(torch.tensor(test_X[:6], dtype = torch.float32), 'test')
            pred_label = torch.argmax(pred, -1)
            print(pred_label)
            for i in range(6):
                optimizer.zero_grad()
                model.zero_grad()
                pred[i][pred_label[i]].backward(retain_graph = True)
                conv2_grad.append(conv2_param.grad.detach().numpy())
                print("pred_label: {} - conv2_param.grad: {}".format(pred_label[i], conv2_param.grad.detach().numpy().shape))
            display(test_X[:6], conv2_grad)

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

    
    # cv2.destroyAllWindows()



# Reference website: https://blog.csdn.net/sinat_37532065/article/details/103362517
# Reference website: https://debuggercafe.com/pytorch-class-activation-map-using-custom-trained-model/