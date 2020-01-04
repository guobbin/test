import random
import numpy as np
import matplotlib.pyplot as plt
bb
a = 1.716
b = 2/3


def sigmoid(x):
    return 2*a/(1+np.exp(-b*x))-a


def d_sigmoid(x):
    return 2*a*b*np.exp(-b*x)/(1+np.exp(-b*x))**2


def net(params, x):
    w1, b1, w2, b2 = params
    x = np.matmul(x, w1)+b1
    x = sigmoid(x)
    x = np.matmul(x, w2)+b2
    return sigmoid(x)


def loss(y_, y):
    y_ = y_/(2*a) + 0.5
    y = y.view(y_.size())
    return np.mean((y-1)*np.log(1-y_) - y*np.log(y_))


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = np.array(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield features[j], labels[j]


def get_grad_loss(params, x, y):
    w1, b1, w2, b2 = params
    z1 = np.matmul(x, w1)+b1
    a1 = sigmoid(z1)
    z2 = np.matmul(a1, w2)+b2
    a2 = sigmoid(z2)

    y_ = a2/(2*a) + 0.5
    y = y.reshape(y_.shape)
    l = np.mean((y - 1) * np.log(1 - y_) - y * np.log(y_))

    dl_dy_ = 1/y.shape[0] * ((y-1)*(-1/(1-y_)) - y*(1/y_))
    dy__da2 = 1/(2*a)
    da2_dz2 = d_sigmoid(z2)

    dl_dz2 = dl_dy_ * dy__da2 * da2_dz2

    dz2_dw2 = a1.T
    dl_db2 = np.sum(dl_dz2, axis=0)
    dl_dw2 = np.matmul(dz2_dw2, dl_dz2)

    dz2_da1 = w2.T
    da1_dz1 = d_sigmoid(z1)

    dl_dz1 = np.matmul(dl_dz2, dz2_da1) * da1_dz1

    dz1_dw1 = x.T
    dl_db1 = np.sum(dl_dz1, axis=0)
    dl_dw1 = np.matmul(dz1_dw1, dl_dz1)

    grads = np.array([dl_dw1, dl_db1, dl_dw2, dl_db2])
    return grads, l


def sgd(params, lr, grads):
    for param, grad in zip(params, grads):
        param -= lr * grad


w1 = 2*(np.random.rand(3, 2)-0.5)
b1 = 2*(np.random.rand(2) - 0.5)

w2 = 2*(np.random.rand(2, 1)-0.5)
b2 = 2*(np.random.rand(1)-0.5)

# w1 = np.array([[0.5],[0.5],[0.5]])
# b1 = np.array([0.5])
#
# w2 = np.array([[-0.5]])
# b2 = np.array([-0.5])

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
x1 = np.array(x1)
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
x2 = np.array(x2)
y1 = np.zeros(10)
y2 = np.ones(10)
features = np.r_[x1, x2]
labels = np.r_[y1, y2]

lr = 0.1
batch_size = 10

params = [w1, b1, w2, b2]
m = 10000
ls = []

for i in range(m):
    for x, y in data_iter(batch_size, features, labels):
        grads, loss = get_grad_loss(params, x, y)
        # params -= lr * grads
        sgd(params, lr, grads)
        ls.append(loss)

plt.plot([i for i in range(len(ls))], np.array(ls), ls, 'b')
plt.show()

