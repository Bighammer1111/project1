import pandas as pd
import numpy as np
from hrvanalysis import get_time_domain_features,remove_outliers,get_frequency_domain_features,get_csi_cvi_features
from vmdpy import VMD
import matplotlib.pyplot as plt
import math
from scipy.signal import hilbert, chirp

tau = 0#噪聲容限
k = 4#分解模態個數
dc = 1 #訊號若無直流成分設爲0，有則為1
init = 1 #初始化w值，為1時均匀分佈產生的隨機數
tol = 1e-7 #誤差大小常數，決定精度與迭代次數
fs = 128

all_feature = pd.DataFrame(columns=['mean_nni','median_nni','range_nni','mean_hr','max_hr','min_hr',
'nni_50','nni_20','csi','cvi' , 'Modified_csi','lf_hf_ratio','total_power','vlf','freq','encode'])#創建空dataframe

file_path = 'C:\\Users\\nemo2\\Documents\\project\\data\\no'+ '14' + ' '+ 'test'+'.csv'
data = pd.read_csv(file_path,skiprows=range(1,500))#刪除資料前段
print(data[' RR'])
rr_interval = data[' RR']
rr_interval = rr_interval.to_numpy()#rr轉numpy
print(rr_interval)
rr_interval = rr_interval[1:len(rr_interval)-1].astype(float)#numpy轉array

zero_index = np.where(rr_interval != 0)
zero_index = np.array(zero_index)[0]
    
rr_interval= rr_interval[rr_interval != 0]
rr_interval= remove_outliers(rr_intervals=rr_interval, low_rri=500, high_rri=1500)#去除極端值
rr_interval = [x for x in rr_interval if math.isnan(x) == False]

loop= len(rr_interval)-10#以8為單位來切割資料
index = list(range(1,loop))#建立index
feature = pd.DataFrame(columns=['mean_nni','median_nni','range_nni','mean_hr','max_hr','min_hr',
'nni_50','nni_20','csi','cvi' , 'Modified_csi','lf_hf_ratio','total_power','vlf','freq','encode'])#創建空dataframe
pd1 = data[' Green Count'].to_numpy()
alpha = len(pd1) * 1.5 #帶寬限制，一般取資料長度1.5~2倍
u,u_hat,omega = VMD(pd1,alpha,tau,k,dc,init,tol)
ppg_signal = np.array(u[1,:] + u[2,:])
signal_len = np.round(len(ppg_signal) / len(rr_interval))
for j in range(1,loop):


# if spo2[i] == 0:
#     break
# feature.spo2[i] = spo2[i]

    loop_data = rr_interval[j:j+10]
    time_domain_features = get_time_domain_features(loop_data)#從rr值取出特徵
    frequency_domain_features = get_frequency_domain_features(loop_data)
    csi_cvi_features = get_csi_cvi_features(loop_data)
    loop_signal = ppg_signal[int(j * signal_len) : int((j+10) * signal_len)]
    analytic_signal = hilbert(loop_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) /
                    (2.0*np.pi) * fs)
    freq = np.mean(instantaneous_frequency)
    # feature.mean_nni[j] = (time_domain_features['mean_nni'] - individual[i-1,0]) / individual[i-1,1]#從得出的feature存入dataframe
    # feature.median_nni[j] = (time_domain_features['median_nni'] - individual[i-1,2]) / individual[i-1,3]
    # feature.range_nni[j] = (time_domain_features['range_nni'] - individual[i-1,4]) / individual[i-1,5]
    # feature.mean_hr[j] = (time_domain_features['mean_hr'] - individual[i-1,6]) / individual[i-1,7]
    # feature.max_hr[j] = (time_domain_features['max_hr'] - individual[i-1,8]) / individual[i-1,9]
    # feature.min_hr[j] = (time_domain_features['min_hr']  - individual[i-1,10]) / individual[i-1,11]
    # feature.nni_50[j] = (time_domain_features['nni_50']  - individual[i-1,18]) / individual[i-1,19]
    # feature.nni_20[j] = (time_domain_features['nni_20']  - individual[i-1,20]) / individual[i-1,21]
    # feature.csi[j] = (csi_cvi_features['csi']  - individual[i-1,12]) / individual[i-1,13]
    # feature.cvi[j] = (csi_cvi_features['cvi'] - individual[i-1,14]) / individual[i-1,15]
    # feature.Modified_csi[j] = (csi_cvi_features['Modified_csi'] - individual[i-1,16]) / individual[i-1,17]
    # feature.lf_hf_ratio[j] = (frequency_domain_features['lf_hf_ratio'] - individual[i-1,22]) / individual[i-1,23]
    # feature.total_power[j] = (frequency_domain_features['total_power'] - individual[i-1,24]) / individual[i-1,25]
    # feature.vlf[j] = (frequency_domain_features['vlf'] - individual[i-1,26]) / individual[i-1,27]
    feature.mean_nni[j] = time_domain_features['mean_nni']
    feature.median_nni[j] = time_domain_features['median_nni'] 
    feature.range_nni[j] = time_domain_features['range_nni'] 
    feature.mean_hr[j] = time_domain_features['mean_hr'] 
    feature.max_hr[j] = time_domain_features['max_hr']
    feature.min_hr[j] = time_domain_features['min_hr'] 
    feature.nni_50[j] = time_domain_features['nni_50'] 
    feature.nni_20[j] = time_domain_features['nni_20'] 
    feature.csi[j] = csi_cvi_features['csi']  
    feature.cvi[j] = csi_cvi_features['cvi']
    feature.Modified_csi[j] = csi_cvi_features['Modified_csi']
    feature.lf_hf_ratio[j] = frequency_domain_features['lf_hf_ratio']
    feature.total_power[j] = frequency_domain_features['total_power']
    feature.vlf[j] = frequency_domain_features['vlf']
    feature.freq[j] = freq

    # def sigmoid(x):#特徵映射
    #     return 1/(1+np.exp(-x))
    # feature = sigmoid(feature)
    

    feature.encode[j] = 0
all_feature = pd.concat([all_feature , feature],ignore_index= True)
all_feature = all_feature.dropna()


