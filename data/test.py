import pandas as pd
import numpy as np
from hrvanalysis import get_time_domain_features,remove_outliers
from vmdpy import VMD
import matplotlib.pyplot as plt

file_path = 'F:\\project\\data\\no7 baseline.csv'
data = pd.read_csv(file_path,skiprows=range(1,1500))#刪除資料前段
pd1 = data[' IR Count'].to_numpy()
alpha = len(pd1) * 1.5 #帶寬限制，一般取資料長度1.5~2倍
tau = 0#噪聲容限
k = 4#分解模態個數
dc = 1 #訊號若無直流成分設爲0，有則為1
init = 1 #初始化w值，為1時均匀分佈產生的隨機數
tol = 1e-7 #誤差大小常數，決定精度與迭代次數
u,u_hat,omega = VMD(pd1,alpha,tau,k,dc,init,tol)

for i in range(k):
   plt.figure()
   plt.subplot(k,1,i+1)
   plt.plot(u[i,:],linewidth=0.2,c='r')
   plt.ylabel('IMF{}'.format(i+1))
plt.show()