# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import special

x = np.linspace(-10, 10,100)
y = np.tanh(x)         #tanh函数
z = special.expit(x)   #sigmoid函数
plt.figure
def sigmoid():
    plt.plot(x, z, color = "red", linewidth = 2, label="sigmoid")
    #plt.legend(loc='upper left')
    plt.title("Sigmoid")
    plt.savefig('/home/allen/dl_grasp/src/data_expend/make_img/img/sigmoid.png')
    plt.show()

def tanh():
    plt.plot(x, y, color = "red", linewidth = 2, label="tanh")
    plt.title("Tanh")
    plt.savefig('/home/allen/dl_grasp/src/data_expend/make_img/img/tanh.png')
    plt.show()

def relu(x):
    return np.maximum(0,x)
 
def plot_relu():
    x=np.arange(-10,10,0.1)
    y=relu(x)
    plt.plot(x,y,color = "red", linewidth = 2, label="relu")
    plt.title("ReLU")
    plt.savefig('/home/allen/dl_grasp/src/data_expend/make_img/img/relu.png')
    plt.show()

def leaky_relu(x):
    a=0.03
    for i in range(len(x)):
        if x[i]>=0:
            x[i]=x[i]
            print(x[i])
        elif x[i]<0:
            x[i]=x[i]*a
            print(x[i])
    return x
 
def plot_leakyrelu():
    x=np.arange(-10,10,0.1)
    y=leaky_relu(x)
    x=np.arange(-10,10,0.1)
    plt.plot(x,y, color = "red", linewidth = 2, label="leakyrelu")
    plt.title("Leaky-ReLU")
    plt.savefig('/home/allen/dl_grasp/src/data_expend/make_img/img/leakyrelu.png')
    plt.show()
def angle_contiune(x):
    for i in range(len(x)):
        if x[i]>=0:
            x[i]=x[i]%360
        elif x[i]<0:
            x[i]=-1*x[i]
            x[i]=x[i]%360
            x[i]=-1*x[i]
    return x
def plot_angle_contiune():
    x=np.arange(-600,600,1)
    y = angle_contiune(x)
    x=np.arange(-600,600,1)
    plt.plot(x,y, color = "red", linewidth = 2, label="angle continue")
    plt.title("Angle change")
    plt.savefig('/home/allen/dl_grasp/src/data_expend/make_img/img/anglecontinue.png')
    plt.show()
def main():
    #sigmoid()
    #tanh()
    #plot_relu()
    #plot_leakyrelu()
    plot_angle_contiune()
if __name__ == "__main__":
    main()
