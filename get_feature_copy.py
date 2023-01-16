import pandas as pd
import numpy as np
from hrvanalysis import get_time_domain_features,remove_outliers,get_frequency_domain_features,get_csi_cvi_features
from vmdpy import VMD
import matplotlib.pyplot as plt
import math

def get_rrfeature(individual,index,emotion_type,encode):

    all_feature = pd.DataFrame(columns=['mean_nni','median_nni','range_nni','mean_hr','max_hr','min_hr',
    'nni_50','nni_20','csi','cvi' , 'Modified_csi','lf_hf_ratio','total_power','vlf','encode'],index=index)#創建空dataframe
    for i in index:
        file_path = 'F:\\project\\data\\no'+ str(i) + ' '+ emotion_type+'.csv'
        data = pd.read_csv(file_path,skiprows=range(1,500))#刪除資料前段

        rr_interval = data[' RR'].to_numpy()#rr轉numpy
        rr_interval = rr_interval[0:len(rr_interval)].astype(float)#numpy轉array

        zero_index = np.where(rr_interval != 0)
        zero_index = np.array(zero_index)[0]
            
        rr_interval= rr_interval[rr_interval != 0]
        rr_interval= remove_outliers(rr_intervals=rr_interval, low_rri=500, high_rri=1500)#去除極端值
        rr_interval = [x for x in rr_interval if math.isnan(x) == False]

        loop= len(rr_interval)-10#以8為單位來切割資料
        index = list(range(1,loop))#建立index
        feature = pd.DataFrame(columns=['mean_nni','median_nni','range_nni','mean_hr','max_hr','min_hr',
    'nni_50','nni_20','csi','cvi' , 'Modified_csi','lf_hf_ratio','total_power','vlf','encode'],index=index)#創建空dataframe
        
        for j in range(1,loop):
        
        
        # if spo2[i] == 0:
        #     break
        # feature.spo2[i] = spo2[i]

            loop_data = rr_interval[j:j+10]
            time_domain_features = get_time_domain_features(loop_data)#從rr值取出特徵
            frequency_domain_features = get_frequency_domain_features(loop_data)
            csi_cvi_features = get_csi_cvi_features(loop_data)
            feature.mean_nni[j] = (time_domain_features['mean_nni'] - individual[i-1,0]) / individual[i-1,1]#從得出的feature存入dataframe
            feature.median_nni[j] = (time_domain_features['median_nni'] - individual[i-1,2]) / individual[i-1,3]
            feature.range_nni[j] = (time_domain_features['range_nni'] - individual[i-1,4]) / individual[i-1,5]
            feature.mean_hr[j] = (time_domain_features['mean_hr'] - individual[i-1,6]) / individual[i-1,7]
            feature.max_hr[j] = (time_domain_features['max_hr'] - individual[i-1,8]) / individual[i-1,9]
            feature.min_hr[j] = (time_domain_features['min_hr']  - individual[i-1,10]) / individual[i-1,11]
            feature.nni_50[j] = (time_domain_features['nni_50']  - individual[i-1,18]) / individual[i-1,19]
            feature.nni_20[j] = (time_domain_features['nni_20']  - individual[i-1,20]) / individual[i-1,21]
            feature.csi[j] = csi_cvi_features['csi'] 
            feature.cvi[j] = csi_cvi_features['cvi']
            feature.Modified_csi[j] = csi_cvi_features['Modified_csi']
            feature.lf_hf_ratio[j] = frequency_domain_features['lf_hf_ratio']
            feature.total_power[j] = frequency_domain_features['total_power']
            feature.vlf[j] = frequency_domain_features['vlf']

            feature.encode[j] = encode 
        all_feature = pd.concat([all_feature , feature],ignore_index= True)
        all_feature = all_feature.dropna()

        
        
    

    print(all_feature)
    del(rr_interval)
    ppg_signal = 0
    return all_feature,ppg_signal
