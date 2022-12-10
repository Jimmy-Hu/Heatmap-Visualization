import torch
import torch.nn as nn
import torch.nn.functional as F

class cnn(torch.nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, 10, 3, padding = 1)
        self.conv2 = nn.Conv2d(10, 20, 3, padding = 1)
        self.pool = nn.MaxPool2d((2, 2))

        self.linear1 = nn.Linear(20 * 7 * 7, 64)
        self.linear2 = nn.Linear(64, output_dim)
        
    def forward(self, x, mode = 'train'):
        out1 = self.pool(F.relu(self.conv1(x)))
        out2 = self.pool(F.relu(self.conv2(out1)))


        B, C, H, W = out2.size()
        out3 = out2.reshape(B, C * H * W)

        out3 = F.relu(self.linear1(out3))
        out4 = F.log_softmax(self.linear2(out3), 1)

        if mode == 'test': return out4, out2
        return out4

    # def features(self):
    #     return self.conv2

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 32, 3)
        self.dropout = nn.Dropout2d(0.25)
        self.fc = nn.Linear(5408, 10)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


if __name__ == '__main__':
    model = cnn(1, 10)
    img = torch.randn(1, 1, 28, 28)
    for param in model.parameters():
        print(param)
    conv1_param = list(model.parameters())[1]

    def extract(g):
        global x1g
        x1g = g

    conv1_param.register_hook(extract)
    pred = model(img)

    # print("pred[0]: {}".format(pred[0]))
    # print("pred: {}".format(pred))
    model.zero_grad()
    index = torch.argmax(pred).item()
    pred[0, index].backward(retain_graph = True)
    print(conv1_param)
    print(conv1_param.grad)
    print('###################################')

    model.zero_grad()
    index = torch.argmax(pred).item()
    pred[0, index].backward(retain_graph = True)
    print(conv1_param)
    print(conv1_param.grad)
    print('###################################')

    model.zero_grad()
    index = torch.argmax(pred).item()
    pred[0, index].backward(retain_graph = True)
    print(conv1_param)
    print(conv1_param.grad)
    print('###################################')
    # for param in model.named_children():
    #     print(param)
    # print(model['conv1'])
    # for param in model.parameters():
    #     print(param)