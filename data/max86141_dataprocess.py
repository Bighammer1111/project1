import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fftpack import fft
from vmdpy import VMD
from hrvanalysis import get_time_domain_features
from hrvanalysis import plot_psd

data = pd.read_csv('MAX86141_20221003_164718.csv', skiprows=6)

pd1 = np.array(data.LEDC1_PD1)#將資料中red ppg取出轉爲array
pd2 = np.array(data.LEDC1_PD2)#ired
pd1 = pd1[~np.isnan(pd1)]#去除空值
pd2 = pd2[~np.isnan(pd2)]




#vmd參數設定
alpha = len(pd1) * 1.5 #帶寬限制，一般取資料長度1.5~2倍
tau = 0#噪聲容限
k = 4#分解模態個數
dc = 1 #訊號若無直流成分設爲0，有則為1
init = 1 #初始化w值，為1時均匀分佈產生的隨機數
tol = 1e-7 #誤差大小常數，決定精度與迭代次數
u,u_hat,omega = VMD(pd1,alpha,tau,k,dc,init,tol)

#得出u峰值的位置，再利用負值搜尋谷值位置
peaks, _ = find_peaks(u[1,:]+ u[2,:], distance=60)#峰值索引，取60個當中的最大值，取imf12的相加效果較好
peaks = np.delete(peaks,0)
pd1_inverse = -u[1,:]-u[2,:]
dips, _ = find_peaks(pd1_inverse, distance=60)
dips = np.delete(dips,0)

#透過pd1、pd2之間的ac/dc計算出血樣值，存到spo2矩陣中
print(len(peaks))
print(len(dips))
spo2 = np.zeros(len(dips) - 1)
for i in range(0,len(dips) - 1):
    spo2_now = ((pd1[peaks[i]] - pd1[dips[i]]) / pd1[dips[i]]) / ((pd2[peaks[i]] - pd2[dips[i]]) / pd2[dips[i]])
    spo2_now = -16.666 * spo2_now * spo2_now + 8.333 * spo2_now + 100
    spo2[i] = spo2_now
print('血氧陣列：')
print(spo2)
print(np.mean(pd1))



#利用diff函數得出每個峰值之間的間距，計算出心率
nn_intervals_list  = np.diff(peaks)*7.8125#將diff轉爲ms單位rr interval

feature=data = pd.DataFrame()#資料df定義
for i in range(0,len(nn_intervals_list)-8):
    loop_heartrate = nn_intervals_list[i:i+8]
    time_domain_features = get_time_domain_features(loop_heartrate)
    time_domain_features=pd.DataFrame(time_domain_features , index=[0])
    data=pd.concat([data,time_domain_features])
print(data)


heartrate = 60 * 1000 / nn_intervals_list #60秒 samplerate128
print('心率陣列：')
print(heartrate)

#顯示原始圖片peak
plt.plot(pd1)
plt.plot(pd2)
plt.plot(peaks, pd1[peaks], "x")
plt.plot(peaks, pd2[peaks], "x")
plt.plot(dips, pd1[dips], "o")
plt.plot(dips, pd2[dips], "o")
plt.show()

#繪出計算后的心率以及血氧
plt.plot(heartrate)
plt.plot(spo2)
plt.show()
for i in range(k):
   plt.figure()
   plt.subplot(k,1,i+1)
   plt.plot(u[i,:],linewidth=0.2,c='r')
   plt.ylabel('IMF{}'.format(i+1))

#for i in range(k):
#    plt.figure()
#    plt.subplot(k,1,i+1)
#    plt.plot(abs(fft(u[i,:])))
#    plt.ylabel('IMF{}'.format(i+1))
plt.show()
print(abs(fft(u[1,:])))

