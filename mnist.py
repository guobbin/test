#!/usr/bin/python
# -*- coding: utf-8 -*-


"""
author ： duanxxnj@163.com
time : 2016-06-04_16-38

这个例子展示了多项式曲线拟合的特性

多项式曲线拟合分为两个步骤：
1、根据多项式的最高次数，对输入特征向量做特征生成
    对原来的每一个特征向量而言，可以生成一个范特蒙德矩阵（ Vandermonde matrix）

    范特蒙德矩阵的尺寸为：[n_samples , n_degree+1]

    其形式为：
    [[1, x_1, x_1 ** 2, x_1 ** 3, ...],
     [1, x_2, x_2 ** 2, x_2 ** 3, ...],
     ...]

2、基于第一步生成的范特蒙德矩阵，直接使用已有线性回归模型，就可以实现多项式回归

这个例子展示了如何基于线性回归模做非线性回归，其实这个也是核函数的基本思想。
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 多项式回归需要拟合的函数
def f(x):
    return x * np.sin(x)


# 产生绘图用的原始数据点
# 这里产生的点的范围比实际拟合所采用的点的范围要宽一些
# 其目的是为了展示当多项式拟合的次数过高时，过拟合的现象
# 过拟合的模型在训练数据范围内，拟合效果非常好
# 在训练数据范围外，模型的拟合效果特别误差
x_plot = np.linspace(-1, 13, 140)

# 训练用数据范围
x = np.linspace(0, 10, 100)

# 随机取训练数据中的10个点作为拟合用的点
rng = np.random.RandomState(0)
rng.shuffle(x)
x = np.sort(x[:10])
y = f(x)

# 将数据从行向量换为列向量，这样每一行就能代表一个样本
X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]

# 从次数为1一直到次数变为17，模型的次数增长步长为3
# 下面会绘制出不同的次数所对应的图像
# 需要注意的是，这6个图的坐标系的y轴的数据范围相差是非常大的
# 模型的次数越高，在训练数据外的测试点上，y的数据和原始数据相差越大
# 即：过拟合现象越明显
#
# 同时，下面还输出了不同次数下，模型对应的参数向量w
# 可以看到，模型次数越大，模型所对应的参数向量的模||w||也越大
# 即：过拟合现象越明显，模型所对应的参数向量的模||w||也越大
#
# 在损失函数后面，加上模型所对应的参数向量的模||w||
# 那么，在最小化损失函数的同时，也限制了参数向量的模||w||的增长
# 这就是正则化可以防止过拟合的原因
#
# 但是在实际测试中发现，如果随机取训练数据的时候，选取的是20个点
# 那么参数向量的模||w||并不是随着模型复杂度的增加而增加
# 这个是因为训练的样本足够大的时候，能够有效的描述原始数据分布
# 那么过拟合的这一套理论就不是特别的适用了
# 所以，方法的选择还是要建立在对数据分布充分的认识上才行
#
for degree in range(9):

    # 基于不同的次数生成多项式模型
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)

    # 不同次数下，多项式模型的参数
    print('模型次数为：', degree, ' 时，模型的参数向量的模：')
    print( np.dot(np.array(model.steps[1][2].coef_),
                 np.array(model.steps[1][3].coef_)))

    print( '模型的参数为：')
    print (model.steps[1][4].coef_)

    y_plot = model.predict(X_plot)

    plt.subplot('52' + str(degree + 1))

    plt.grid()
    plt.plot(x_plot, f(x_plot), label="ground truth")
    plt.scatter(x, y, label="training points")
    plt.plot(x_plot, y_plot, label="degree %d" % degree)
    plt.legend(loc='lower left')

plt.show()