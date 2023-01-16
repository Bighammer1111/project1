import pandas as pd
import numpy as np
from hrvanalysis import get_time_domain_features,remove_outliers
from vmdpy import VMD
import matplotlib.pyplot as plt
import math

def get_rrfeature(index,emotion_type,encode):
    
    
    tau = 0#噪聲容限
    k = 4#分解模態個數
    dc = 1 #訊號若無直流成分設爲0，有則為1
    init = 1 #初始化w值，為1時均匀分佈產生的隨機數
    tol = 1e-7 #誤差大小常數，決定精度與迭代次數
    for i in index:
        file_path = 'F:\\project\\data\\no'+ str(i) + ' '+ emotion_type+'.csv'
        data = pd.read_csv(file_path,skiprows=range(1,1000))#刪除資料前段
        pd1 = data[' Green Count'].to_numpy()
        alpha = len(pd1) * 1.5 #帶寬限制，一般取資料長度1.5~2倍
        u,u_hat,omega = VMD(pd1,alpha,tau,k,dc,init,tol)

        if(i == 1):
            neual_all = data
            ppg_signal = np.array(u[1,:] + u[2,:])
        else:
            neual_all=pd.concat([neual_all,data], ignore_index=True)
            reg = np.array(u[1,:] + u[2,:])
            ppg_signal = np.concatenate ((ppg_signal,reg))

    rr_interval = neual_all[' RR'].to_numpy()#rr轉numpy
    rr_interval = rr_interval[0:len(rr_interval)].astype(float)#numpy轉array

    zero_index = np.where(rr_interval != 0)
    zero_index = np.array(zero_index)[0]
            
    rr_interval= rr_interval[rr_interval != 0]
    spo2 = neual_all[' SpO2 (%)'][zero_index].to_numpy()
    spo2 = spo2[0:len(spo2)].astype(float)#numpy轉array 
    spo2 = spo2[spo2 != 0]  

    rr_interval= remove_outliers(rr_intervals=rr_interval, low_rri=500, high_rri=1500)#去除極端值
    rr_interval = [x for x in rr_interval if math.isnan(x) == False]


    loop= round(len(rr_interval)/8)#以8為單位來切割資料
    index = list(range(1,loop-8))#建立index
    feature = pd.DataFrame(columns=['mean_nni','median_nni','range_nni','mean_hr','max_hr','min_hr','encode'],index=index)#創建空dataframe
    for i in range(1,loop):
        
        
        # if spo2[i] == 0:
        #     break
        # feature.spo2[i] = spo2[i]

        loop_data = rr_interval[i:i+8]
        time_domain_features = get_time_domain_features(loop_data)#從rr值取出特徵
        feature.mean_nni[i] = time_domain_features['mean_nni'] #從得出的feature存入dataframe
        feature.median_nni[i] = time_domain_features['median_nni'] 
        feature.range_nni[i] = time_domain_features['range_nni'] 
        feature.mean_hr[i] = time_domain_features['mean_hr'] 
        feature.max_hr[i] = time_domain_features['max_hr'] 
        feature.min_hr[i] = time_domain_features['min_hr']  

        feature.encode[i] = encode
    # feature['mean_nni'] = (feature['mean_nni'] - feature['mean_nni'].mean() )/ feature['mean_nni'].std()
    # feature['median_nni'] = (feature['median_nni'] - feature['median_nni'].mean() )/ feature['median_nni'].std()
    # feature['mean_hr'] = (feature['mean_hr'] - feature['mean_hr'].mean() )/ feature['mean_hr'].std()
    # feature['max_hr'] = (feature['max_hr'] - feature['max_hr'].mean() )/ feature['max_hr'].std()
    # feature['min_hr'] = (feature['min_hr'] - feature['min_hr'].mean() )/ feature['min_hr'].std()
    return feature,ppg_signal

def get_1rrfeature(index,emotion_type,encode):
    
    
    tau = 0#噪聲容限
    k = 4#分解模態個數
    dc = 1 #訊號若無直流成分設爲0，有則為1
    init = 1 #初始化w值，為1時均匀分佈產生的隨機數
    tol = 1e-7 #誤差大小常數，決定精度與迭代次數
    for i in index:
        file_path = 'F:\\project\\data\\no'+ str(i) + ' '+ emotion_type+'.csv'
        data = pd.read_csv(file_path,skiprows=range(1,1000))#刪除資料前段
        pd1 = data[' Green Count'].to_numpy()
        alpha = len(pd1) * 1.5 #帶寬限制，一般取資料長度1.5~2倍
        u,u_hat,omega = VMD(pd1,alpha,tau,k,dc,init,tol)


        neual_all = data
        ppg_signal = np.array(u[1,:] + u[2,:])


    rr_interval = neual_all[' RR'].to_numpy()#rr轉numpy
    rr_interval = rr_interval[0:len(rr_interval)].astype(float)#numpy轉array

    zero_index = np.where(rr_interval != 0)
    zero_index = np.array(zero_index)[0]
            
    rr_interval= rr_interval[rr_interval != 0]
    spo2 = neual_all[' SpO2 (%)'][zero_index].to_numpy()
    spo2 = spo2[0:len(spo2)].astype(float)#numpy轉array   

    rr_interval= remove_outliers(rr_intervals=rr_interval, low_rri=500, high_rri=1500)#去除極端值
    rr_interval = [x for x in rr_interval if math.isnan(x) == False]


    loop= round(len(rr_interval)/8)#以8為單位來切割資料
    index = list(range(1,loop-8))#建立index
    feature = pd.DataFrame(columns=['mean_nni','median_nni','range_nni','mean_hr','max_hr','min_hr','encode'],index=index)#創建空dataframe
    for i in range(1,loop):
        # if spo2[i] == 0:
        #     break
        # feature.spo2[i] = spo2[i]
        
        loop_data = rr_interval[i:i+8]
        time_domain_features = get_time_domain_features(loop_data)#從rr值取出特徵
        feature.mean_nni[i] = time_domain_features['mean_nni'] #從得出的feature存入dataframe
        feature.median_nni[i] = time_domain_features['median_nni'] 
        feature.range_nni[i] = time_domain_features['range_nni'] 
        feature.mean_hr[i] = time_domain_features['mean_hr'] 
        feature.max_hr[i] = time_domain_features['max_hr'] 
        feature.min_hr[i] = time_domain_features['min_hr']  

        


        feature.encode[i] = encode
    # feature['mean_nni'] = (feature['mean_nni'] - feature['mean_nni'].mean() )/ feature['mean_nni'].std()
    # feature['median_nni'] = (feature['median_nni'] - feature['median_nni'].mean() )/ feature['median_nni'].std()
    # feature['mean_hr'] = (feature['mean_hr'] - feature['mean_hr'].mean() )/ feature['mean_hr'].std()
    # feature['max_hr'] = (feature['max_hr'] - feature['max_hr'].mean() )/ feature['max_hr'].std()
    # feature['min_hr'] = (feature['min_hr'] - feature['min_hr'].mean() )/ feature['min_hr'].std()
    # plt.plot(u[0,:])
    # plt.show()
    # plt.plot(u[1,:])
    # plt.show()
    # plt.plot(u[2,:])
    # plt.show()
    # plt.plot(u[3,:])
    # plt.show()
    # plt.plot(u[4,:])
    # plt.show()
    return feature,ppg_signal
