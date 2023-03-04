import argparse
import math
import random

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--task", default="",
#                         help="this is help")
#     parser.add_argument("--dataset", default="nuscenes",
#                         help="this is dataset")
#     parser.add_argument("--x", type=float, default=1.0)
#     parser.add_argument("--t", action="store_true",
#                         help="test true")
#     opt = parser.parse_args()
#     print(type(opt.x))


#x_i = math.sin(x_i_1) + 5 * x_i_1/(x_i_1**2+1)+Q
#y_i = x_i **2 +R
N = 100
x = list(range(N))
y = list(range(N))
x[0] = 0.1
y[0] = 0.01**2
for i in range(1,N):
    x[i] = math.sin(x[i-1]) + 5 * x[i-1]/(x[i-1]**2+1)
    y[i] = x[i]**2+random.gauss(0, 1)
    
X_old = list(range(N))
X_new = list(range(N))
X_plus = list(range(N)) #用于存放滤波值，每次迭代后，计算后验概率的期望值
w = list(range(N)) #存放权重
#设置初始值X0（i），如果对于初值有自信，可以让所有粒子都相同
for i in range(0,N):
    X_old[i] = 0.1
    w[i] = 1/N

for i in range(1,N):
    #预测步：从X0推出X1
    for j in range(0,N):
        X_old[j] = math.sin(X_old[j]) + 5 * X_old[j]/(X_old[j]**2+1)+random.gauss(0, 1)
    #预测步完成
    
    #更新步
    for j in range(0,N):
        #w[j] = w[j] * fR(...)
        #fR(...) = (2 * math.pi *R) ** (-0.5) * math.exp(-((y[i] - x_old[j]**2)**2/(2*R))) 
        w[j] = math.exp(-((y[i] - X_old[j]**2)**2/(2*0.001)))  
    #更新步完成
    
    #归一化
    for j in range(0,N):
        w[j] = w[j] / sum(w)
    #因为k*w / sum（k*w）结果一样，所以常数不会影响归一化结果，因此，更新步更简洁
    #若不是每一次都重采样，则更新步w[j]就要相应修改，加上乘以w[j]
    
    #重采样(可以每次都重采样，也可以当粒子数低于某一阈值再采样N<1/sum(w**2))
    c = list(range(N)) #将w[j]按权重划分区间
    c[0] = w[0]
    for j in range(1,N):
        c[j] = c[j-1] + w[j]
    #生产随机数，看落在C的哪个区间
    #重采样数量为N个粒子，与之前相同
    for j in range(0,N):
        a = random.uniform(0,1) #生成随机数a
        for k in range(0,N):
            if a<c[k]:
                X_new[j] = X_old[k]
                break #一定要break，否则重采样粒子会被后续覆盖
    #重采样完毕
    
    #将新的粒子赋值给X_old,为下一步递推做准备
    X_old = X_new
    #将权重设为1/N
    for j in range(0,N):
        w[j] = 1/N
    #将每一步后验概率的期望值赋予X_plus,即通过滤波算法得到的值
    X_plus[i] = sum(X_new)/N