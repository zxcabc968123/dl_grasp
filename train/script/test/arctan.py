import numpy as np
import pandas as pd
import math
import time

def arc_cal(cos_x,sin_x):
    result = 0.5*np.arctan (sin_x/cos_x)
    result = result/math.pi*180
    print('arc_result :{}'.format(result))
    print('cos_2x : {} sin_2x : {}'.format(math.cos(2*result*math.pi/180),math.sin(2*result*math.pi/180)))
def arc_input(x):
    print(x)
    x=x*math.pi/180
    print(x)
    cos_x = math.cos(2*x)
    sin_x = math.sin(2*x)
    print('cos_x :',cos_x)
    print('sin_x :',sin_x)
    return (cos_x,sin_x)
def sincos_90(degree):
    degree_sin = math.sin(degree*math.pi/180)
    degree_cos = math.cos(degree*math.pi/180)

    return (degree_sin,degree_cos)

def main():
    (cos_x,sin_x)=arc_input(30)
    result = 0.5*np.arctan (sin_x/cos_x)
    #result = 0.5*np.arctan (cos_x/cos_x)
    result = round(result/math.pi*180)
    print('arctan_result = {}'.format(result))

    (x_degree_sin,x_degree_cos) = sincos_90(-60)

    (y_degree_sin,y_degree_cos) = sincos_90(33.4155)

    
    print('x_degree_cos :',x_degree_cos)
    print('x_degree_sin :',x_degree_sin)
    
    print('y_degree_cos :',y_degree_cos)
    print('y_degree_sin :',y_degree_sin)


    print('x_degree_sin*x_degree_cos :',x_degree_sin*x_degree_cos)
    print('y_degree_sin*y_degree_cos :',y_degree_sin*y_degree_cos)
    #print('x_degree_sin*x_degree_cos :',x_degree_sin*x_degree_cos)

    arc_cal(-0.49999999,-0.86602)
    #arc_cal()


    
if __name__ == "__main__":
    main()