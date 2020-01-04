import torch
import random
import numpy as np
import matplotlib.pyplot as plt
# torch.manual_seed(1)
# np.random.seed(12)
# random.seed(123)
a = 1.716
b = 2/3

def sigmoid(x):
    return 2*a/(1+torch.exp(-b*x))-a

def net(params, x):
    w1, b1, w2, b2 = params
    x = torch.mm(x, w1)+b1
    x = sigmoid(x)
    x = torch.mm(x, w2)+b2
    return sigmoid(x)

def loss(y_, y):
    y_ = y_/(2*a) + 0.5
    y = y.view(y_.size())
    return torch.mean((y-1)*torch.log(1-y_) - y*torch.log(y_))

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield  features.index_select(0, j), labels.index_select(0, j)

def sgd(params, lr, batch_size):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    for param in params:
        print(param.grad)
        param.data -= lr * param.grad # 注意这里更改param时用的param.data


# w1 = 2*(torch.rand(3,1)-0.5)
# b1 = 2*(torch.rand(1) - 0.5)
#
# w1 = 2*(torch.rand(3,1)-0.5)
# b1 = 2*(torch.rand(1) - 0.5)
#
# w2 = 2*(torch.rand(1,1)-0.5)
# b2 = 2*(torch.rand(1)-0.5)

w1 = torch.tensor([[0.5],[0.5],[0.5]])
b1 = torch.tensor([0.5])

w2 = torch.tensor([[-0.5]])
b2 = torch.tensor([-0.5])
# w1 = torch.zeros(3,5)
# b1 = torch.zeros(2)
# b2 = 2*(torch.rand(1) - 0.5)
# b2 = torch.zeros(1)

# w2 = torch.zeros(2,1)

x1 = [[0.28, 1.31, -6.2],
      [0.07, 0.58, -0.78],
      [1.54, 2.01, -1.63],
      [-0.44, 1.18, -4.32],
      [-0.81, 0.21, 5.73],
      [1.52, 3.16, 2.77],
      [2.20, 2.42, -0.19],
      [0.91, 1.94, 6.21],
      [0.65, 1.93, 4.38],
      [-0.26, 0.82, -0.96]]
x1 = torch.tensor(x1)
x2 = [[0.011, 1.03, -0.21],
      [1.27, 1.28, 0.08],
      [0.13, 3.12, 0.16],
      [-0.21, 1.23, -0.11],
      [-2.18, 1.39, -0.19],
      [0.34, 1.96, -0.16],
      [-1.38, 0.94, 0.45],
      [-0.12, 0.82, 0.17],
      [-1.44, 2.31, 0.14],
      [0.26, 1.94, 0.08]]
x2 = torch.tensor(x2)
y1 = torch.zeros(10)
y2 = torch.ones(10)
features = torch.cat((x1,x2),axis=0)
labels = torch.cat((y1,y2),axis=0)

lr = 0.1
batch_size = 20
# next(data_iter(batch_size, features, labels))
params = [w1, b1, w2, b2]
m = 1000
ls = []
for param in params:
    param.requires_grad_(requires_grad=True)


for i in range(m):
    for x, y in data_iter(batch_size, features, labels):
        y_ = net(params, x)
        l = loss(y_, y)
        ls.append(l)
        l.backward()
        sgd(params, lr, batch_size)
    for param in params:
#        print(param.grad.data)
        param.grad.data.zero_()


plt.plot([i for i in range(len(ls))], np.array(ls), ls, 'b')
plt.show()
