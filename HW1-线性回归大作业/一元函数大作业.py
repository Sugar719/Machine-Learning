# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 12:37:19 2021

@author: jsx20
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import random
from sklearn.linear_model import Ridge, RidgeCV


train_data_number = 20


def f(x):
    return 1/(1+25*x**2)

x = np.arange(-1,1,0.01)
y = f(x)

plt.figure()
plt.ylim(-0.5,1.5)
plt.plot(x,y,label = "target function")
plt.xlabel("x")
plt.ylabel("y")

#训练集
a_train = np.random.random((1, train_data_number)) - 0.5
x_train = np.linspace(-1, 1, train_data_number)
y_train = f(x_train)
y_train = y_train - a_train*0.5
y_train = y_train[0]

#测试集
a_test = np.random.random((1, 30)) - 0.5
x_test = np.linspace(-1.0, 1.0, 30)
y_test = f(x_test)
y_test = y_test - a_test * 0
y_test = y_test[0]


x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

plt.scatter(x_train, y_train)

#多项式回归
n = 0
error_test = []
error_train = []
for i in range(15,16):
    reg = PolynomialFeatures(degree = i)
    x_train_quadratic = reg.fit_transform(x_train)
    model = LinearRegression()
    model = model.fit(x_train_quadratic,y_train)
    
    #画出拟合函数
    xx = np.linspace(-1, 1, 1000)
    xx_quadratic = reg.transform(xx.reshape(xx.shape[0], 1))#shape[0]为读取第一行长度
    yy_predict = model.predict(xx_quadratic)
    plt.plot(xx, yy_predict, 'r-')
    
    #计算拟合误差
    x_pred_train = reg.transform(x_train.reshape(x_train.shape[0],1))
    y_pred_train = model.predict(x_pred_train)
    error_train.append(mean_squared_error(y_train,y_pred_train))
    x_pred_test = reg.transform(x_test.reshape(x_test.shape[0],1))
    y_pred_test = model.predict(x_pred_test)
    error_test.append(mean_squared_error(y_test,y_pred_test))
    print(i,'次多项式回归方差为：','训练集',mean_squared_error(y_train,y_pred_train),'测试集',mean_squared_error(y_test,y_pred_test))
    
    
# print("测试集最小",error_test.index(min(error_test))+1,"训练集最小",error_train.index(min(error_train))+1) 
    


for i in [0, 0.0001,0.01, 0.1, 1]:
    Ridge_poly9 = Ridge(alpha=i)#创建模型
#     Ridge_poly9 = RidgeCV() #多个alphas,可以得到最佳alphas和w值      
    Ridge_poly9.fit(x_train_quadratic, y_train)#进行拟合
    y_Ridge_pred_test = Ridge_poly9.predict(x_pred_test)
    yy_Ridge_pred = Ridge_poly9.predict(xx_quadratic)
    plt.plot(xx, yy_Ridge_pred)
    print('系数为%f时' % i)
    print("L2正则化的均方根误差：", mean_squared_error(y_test, y_Ridge_pred_test))
    
    
    
    
    
    
    
    
    
    
    