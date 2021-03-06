# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 13:44:08 2021

@author: jsx20
"""

import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV 

def Plot():
    #画图
    #画图部分
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_zlim3d(-1, 1)
    
    # Make data.
    X1 = np.arange(-2, 2, 0.1)
    X2 = np.arange(-2, 2, 0.1)
    X1, X2 = np.meshgrid(X1,X2)
    R1 = 1+np.sin(2*X1+3*X2)
    R2 = 3.5+np.sin(X1-X2)
    Y = R1/R2
    
    # Plot the surface.
    surf = ax.plot_surface(X1, X2, Y, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    
    
    ax = fig.gca(projection='3d')
    x1 = np.arange(-2, 2, 0.1)
    x2 = np.arange(-2, 2, 0.1)
    x1,x2 = np.meshgrid(x1,x2)
    X = np.hstack((x1.reshape(-1,1),x2.reshape(-1,1)))
    x = poly.fit_transform(X)
    y = reg.predict(x)
    ax.scatter(x1, x2, y, c = 'r', marker = 'o') 



#创建训练集
x1_train = np.arange(-2,2,0.05)
x2_train = np.arange(-2,2,0.05)
R1 = 1+np.sin(2*x1_train+3*x2_train)
R2 = 3.5+np.sin(x1_train-x2_train)
y_train = R1/R2
X_train = np.hstack((x1_train.reshape(-1,1),x2_train.reshape(-1,1)))

#创建测试集
x1_test = np.arange(-2,2,0.01)
x2_test = np.arange(-2,2,0.01)
R1 = 1+np.sin(2*x1_test+3*x2_test)
R2 = 3.5+np.sin(x1_test-x2_test)
y_test = R1/R2
X_test = np.hstack((x1_test.reshape(-1,1),x2_test.reshape(-1,1)))

#进行拟合
for i in range(2,30):
    poly = PolynomialFeatures(degree=i)#多项式拟合
    # reg = linear_model.Ridge(alpha = 0) 
    reg = linear_model.LinearRegression()#创建一个线性回归
    X_train_quadratic = poly.fit_transform(X_train)#把训练好的x值的多项式特征实例应用到一系列点上，形成矩阵
    reg = reg.fit(X_train_quadratic,y_train)# 进行训练
    
    
    #计算损失
    x_pred1 = poly.fit_transform(X_test)
    y_pred1 = reg.predict(x_pred1)
    model_test_mean_sqaured_error = mean_squared_error(y_test, y_pred1)#测试集误差
    
    x_pred2 = poly.fit_transform(X_train)
    y_pred2 = reg.predict(x_pred2)
    model_train_mean_sqaured_error = mean_squared_error(y_train, y_pred2)#训练集误差
    print("多项式次数为:{}时，训练集方差为:{}".format(i,model_train_mean_sqaured_error),end=" ")
    print("测试集方差为:{}".format(model_test_mean_sqaured_error))
        
    
    for j in [0, 0.0001,0.01, 0.1, 1]:
        Ridge_poly = linear_model.Ridge (alpha = j)#创建模型（正则化）
        #     Ridge_poly9 = RidgeCV() #多个alphas,可以得到最佳alphas和w值      
        Ridge_poly = Ridge_poly.fit(X_train_quadratic, y_train)#进行拟合
        
        xx_pred1 = poly.fit_transform(X_test)
        yy_Ridge_pred1 = Ridge_poly.predict(xx_pred1)#测试集预测
        
        xx_pred2 = poly.fit_transform(X_train)
        yy_Ridge_pred2 = Ridge_poly.predict(xx_pred2)#训练集预测
        
        print('系数为%f时' % j)
        print("L2正则化的测试集均方根误差：", mean_squared_error(y_test, yy_Ridge_pred1))
        print()
        #   print('最佳的alpha：',Ridge_poly9.alpha_) # 只有在使用RidgeCV时才有效
    
        

#Plot()







