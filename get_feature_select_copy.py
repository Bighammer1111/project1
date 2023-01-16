import pandas as pd
import numpy as np
from hrvanalysis import get_time_domain_features,remove_outliers
from vmdpy import VMD
import matplotlib.pyplot as plt
import math

def get_rrfeature(individual,index,emotion_type,encode):
    

    
    tau = 0#噪聲容限
    k = 4#分解模態個數
    dc = 1 #訊號若無直流成分設爲0，有則為1
    init = 1 #初始化w值，為1時均匀分佈產生的隨機數
    tol = 1e-7 #誤差大小常數，決定精度與迭代次數
    all_feature = pd.DataFrame(columns=['mean_nni','median_nni','range_nni','mean_hr','max_hr','min_hr','encode'],index=index)#創建空dataframe
    for i in index:
        file_path = 'F:\\project\\data\\no'+ str(i) + ' '+ emotion_type+'.csv'
        data = pd.read_csv(file_path,skiprows=range(1,200))#刪除資料前段
        pd1 = data[' Green Count'].to_numpy()
        alpha = len(pd1) * 1.5 #帶寬限制，一般取資料長度1.5~2倍
        u,u_hat,omega = VMD(pd1,alpha,tau,k,dc,init,tol)
        rr_interval = data[' RR'].to_numpy()#rr轉numpy
        rr_interval = rr_interval[0:len(rr_interval)].astype(float)#numpy轉array

        zero_index = np.where(rr_interval != 0)
        zero_index = np.array(zero_index)[0]
            
        rr_interval= rr_interval[rr_interval != 0]
        rr_interval= remove_outliers(rr_intervals=rr_interval, low_rri=500, high_rri=1500)#去除極端值
        rr_interval = [x for x in rr_interval if math.isnan(x) == False]

        loop= len(rr_interval)-10#以8為單位來切割資料
        index = list(range(1,loop))#建立index
        feature = pd.DataFrame(columns=['mean_nni','median_nni','range_nni','mean_hr','max_hr','min_hr','encode'],index=index)#創建空dataframe
        
        for j in range(1,loop):
        
        
        # if spo2[i] == 0:
        #     break
        # feature.spo2[i] = spo2[i]

            loop_data = rr_interval[j:j+10]
            time_domain_features = get_time_domain_features(loop_data)#從rr值取出特徵
            feature.mean_nni[j] = (time_domain_features['mean_nni'] - individual[i-1,0]) / individual[i-1,1]#從得出的feature存入dataframe
            feature.median_nni[j] = (time_domain_features['median_nni'] - individual[i-1,2]) / individual[i-1,3]
            feature.range_nni[j] = (time_domain_features['range_nni'] - individual[i-1,4]) / individual[i-1,5]
            feature.mean_hr[j] = (time_domain_features['mean_hr'] - individual[i-1,6]) / individual[i-1,7]
            feature.max_hr[j] = (time_domain_features['max_hr'] - individual[i-1,8]) / individual[i-1,9]
            feature.min_hr[j] = (time_domain_features['min_hr']  - individual[i-1,10]) / individual[i-1,11]

            feature.encode[j] = encode 
        all_feature = pd.concat([all_feature , feature],ignore_index= True)
        all_feature = all_feature.dropna()

        
        
    del(rr_interval)
    ppg_signal = 0
    return all_feature,ppg_signal

# def get_1rrfeature(individual,index,emotion_type,encode):
    
    
#     tau = 0#噪聲容限
#     k = 4#分解模態個數
#     dc = 1 #訊號若無直流成分設爲0，有則為1
#     init = 1 #初始化w值，為1時均匀分佈產生的隨機數
#     tol = 1e-7 #誤差大小常數，決定精度與迭代次數
#     for i in index:
#         file_path = 'F:\\project\\data\\no'+ str(i) + ' '+ emotion_type+'.csv'
#         data = pd.read_csv(file_path,skiprows=range(1,1000))#刪除資料前段
#         pd1 = data[' Green Count'].to_numpy()
#         alpha = len(pd1) * 1.5 #帶寬限制，一般取資料長度1.5~2倍
#         u,u_hat,omega = VMD(pd1,alpha,tau,k,dc,init,tol)
#         rr_interval = data[' RR'].to_numpy()#rr轉numpy
#         rr_interval = rr_interval[0:len(rr_interval)].astype(float)#numpy轉array

#         zero_index = np.where(rr_interval != 0)
#         zero_index = np.array(zero_index)[0]
            
#         rr_interval= rr_interval[rr_interval != 0]
#         rr_interval= remove_outliers(rr_intervals=rr_interval, low_rri=500, high_rri=1500)#去除極端值
#         rr_interval = [x for x in rr_interval if math.isnan(x) == False]

#         loop= round(len(rr_interval)/8)#以8為單位來切割資料
#         index = list(range(1,loop-8))#建立index
#         feature = pd.DataFrame(columns=['mean_nni','median_nni','range_nni','mean_hr','max_hr','min_hr','encode'],index=index)#創建空dataframe
        
#         for j in range(1,loop):
        
        
#         # if spo2[i] == 0:
#         #     break
#         # feature.spo2[i] = spo2[i]

#             loop_data = rr_interval[j:j+8]
#             time_domain_features = get_time_domain_features(loop_data)#從rr值取出特徵
#             feature.mean_nni[j] = (time_domain_features['mean_nni'] - individual[i-1,0]) / individual[i-1,1]#從得出的feature存入dataframe
#             feature.median_nni[j] = (time_domain_features['median_nni'] - individual[i-1,2]) / individual[i-1,3]
#             feature.range_nni[j] = (time_domain_features['range_nni'] - individual[i-1,4]) / individual[i-1,5]
#             feature.mean_hr[j] = (time_domain_features['mean_hr'] - individual[i-1,6]) / individual[i-1,7]
#             feature.max_hr[j] = (time_domain_features['max_hr'] - individual[i-1,8]) / individual[i-1,9]
#             feature.min_hr[j] = (time_domain_features['min_hr']  - individual[i-1,10]) / individual[i-1,11]

#             feature.encode[j] = encode 
        
#         data = pd.DataFrame(columns=['mean_nni','median_nni','range_nni','mean_hr','max_hr','min_hr','encode'],index=index)#創建空dataframe
#         data = pd.concat([data , feature])




#     ppg_signal = 0
#     return feature,ppg_signal
