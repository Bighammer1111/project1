import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from hrvanalysis import get_time_domain_features,remove_outliers,get_frequency_domain_features,get_csi_cvi_features

from get_feature_copy import get_rrfeature
import math

all_index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
test_neual_index=[13]
test_positive_index = [13]
test_negetive_index = [13]
neual_index=[1,2,3,4,5,6,7,8,9,11,12,10,14]
positive_index = [1,2,3,4,5,6,7,8,9,11,12,10,14]
negetive_index = [1,2,3,4,5,6,7,8,9,11,12,10,14]
neual_index = np.delete(neual_index,np.where(neual_index == 10))
positive_index = np.delete(positive_index,np.where(neual_index == 10))
negetive_index = np.delete(negetive_index,np.where(neual_index == 10))#將測試資料從訓練資料中刪除

print(test_neual_index)
print(neual_index)
# test_neual_index=[14]
# test_positive_index = [14]
# test_negetive_index = [14]
# neual_index=[14]
# positive_index = [14]
# negetive_index = [14]


mean_nni_mean = np.zeros((all_index[-1] , 1))
mean_nni_std = np.zeros((all_index[-1] , 1))
median_nni_mean = np.zeros((all_index[-1] , 1))
median_nni_std = np.zeros((all_index[-1] , 1))
range_nni_mean = np.zeros((all_index[-1] , 1))
range_nni_std = np.zeros((all_index[-1] , 1))
mean_hr_mean = np.zeros((all_index[-1] , 1))
mean_hr_std = np.zeros((all_index[-1] , 1))
max_hr_mean = np.zeros((all_index[-1] , 1))
max_hr_std = np.zeros((all_index[-1] , 1))
min_hr_mean = np.zeros((all_index[-1] , 1))
min_hr_std = np.zeros((all_index[-1] , 1))
nni_50_mean = np.zeros((all_index[-1] , 1))
nni_50_std = np.zeros((all_index[-1] , 1))
nni_20_mean = np.zeros((all_index[-1] , 1))
nni_20_std = np.zeros((all_index[-1] , 1))
csi_mean = np.zeros((all_index[-1] , 1))
csi_std = np.zeros((all_index[-1] , 1))
cvi_mean = np.zeros((all_index[-1] , 1))
cvi_std = np.zeros((all_index[-1] , 1))
Modified_csi_mean = np.zeros((all_index[-1] , 1))
Modified_csi_std = np.zeros((all_index[-1] , 1))
lf_hf_ratio_mean = np.zeros((all_index[-1] , 1))
lf_hf_ratio_std = np.zeros((all_index[-1] , 1))
total_power_mean = np.zeros((all_index[-1] , 1))
total_power_std = np.zeros((all_index[-1] , 1))
vlf_mean = np.zeros((all_index[-1] , 1))
vlf_std = np.zeros((all_index[-1] , 1))

for i in all_index:
        file_path = 'F:\\project\\data\\no'+ str(i) + ' baseline.csv'
        data = pd.read_csv(file_path,skiprows=range(1,1000))#刪除資料前段
        rr_interval = data[' RR'].to_numpy()#rr轉numpy
        rr_interval = rr_interval[0:len(rr_interval)].astype(float)#numpy轉array
        

        zero_index = np.where(rr_interval != 0)
        zero_index = np.array(zero_index)[0]
            
        rr_interval= rr_interval[rr_interval != 0]
        rr_interval= remove_outliers(rr_interval, low_rri=500, high_rri=1500)#去除極端值
        rr_interval = [x for x in rr_interval if math.isnan(x) == False]

        loop= len(rr_interval)-11#以8為單位來切割資料
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
            feature.mean_nni[j] = time_domain_features['mean_nni'] 
            feature.median_nni[j] = time_domain_features['median_nni'] 
            feature.range_nni[j] = time_domain_features['range_nni']
            feature.mean_hr[j] = time_domain_features['mean_hr']
            feature.max_hr[j] = time_domain_features['max_hr'] 
            feature.min_hr[j] =time_domain_features['min_hr']  
            feature.nni_50[j] = time_domain_features['nni_50'] 
            feature.nni_20[j] = time_domain_features['nni_20'] 
            feature.csi[j] = csi_cvi_features['csi']
            feature.cvi[j] = csi_cvi_features['cvi']
            feature.Modified_csi[j] = csi_cvi_features['Modified_csi']
            feature.lf_hf_ratio[j] = frequency_domain_features['lf_hf_ratio']
            feature.total_power[j] = frequency_domain_features['total_power']
            feature.vlf[j] = frequency_domain_features['vlf']


            
        mean_nni_mean[i-1 , 0] =  feature.mean_nni.mean()
        mean_nni_std[i-1 , 0] =  feature.mean_nni.std()
        median_nni_mean[i-1 , 0] =  feature.median_nni.mean()
        median_nni_std[i-1 , 0] =  feature.median_nni.std()
        range_nni_mean[i-1 , 0] =  feature.range_nni.mean()
        range_nni_std[i-1 , 0] =  feature.range_nni.std()
        mean_hr_mean[i-1 , 0] =  feature.mean_hr.mean()
        mean_hr_std[i-1 , 0] =  feature.mean_hr.std()
        max_hr_mean[i-1 , 0] =  feature.max_hr.mean()
        max_hr_std[i-1 , 0] =  feature.max_hr.std()
        min_hr_mean[i-1 , 0] =  feature.min_hr.mean()
        min_hr_std[i-1 , 0] =  feature.min_hr.std()
        nni_50_mean[i-1 , 0] =  feature.nni_50.mean()
        nni_50_std[i-1 , 0] =  feature.nni_50.std()
        nni_20_mean[i-1 , 0] =  feature.nni_20.mean()
        nni_20_std[i-1 , 0] =  feature.nni_20.std()
        csi_mean[i-1 , 0] =  feature.csi.mean()
        csi_std[i-1 , 0] =  feature.csi.std()
        cvi_mean[i-1 , 0] =  feature.cvi.mean()
        cvi_std[i-1 , 0] =  feature.cvi.std()
        Modified_csi_mean[i-1 , 0] =  feature.Modified_csi.mean()
        Modified_csi_std[i-1 , 0] =  feature.Modified_csi.std()
        lf_hf_ratio_mean[i-1 , 0] =  feature.lf_hf_ratio.mean()
        lf_hf_ratio_std[i-1 , 0] =  feature.lf_hf_ratio.std()
        total_power_mean[i-1 , 0] =  feature.total_power.mean()
        total_power_std[i-1 , 0] =  feature.total_power.std()
        vlf_mean[i-1 , 0] =  feature.vlf.mean()
        vlf_std[i-1 , 0] =  feature.vlf.std()

        individual = np.concatenate((mean_nni_mean , mean_nni_std , median_nni_mean , median_nni_std , range_nni_mean , range_nni_std , mean_hr_mean , mean_hr_std , max_hr_mean , max_hr_std , min_hr_mean , min_hr_std
         , csi_mean , csi_std , cvi_mean , cvi_std , Modified_csi_mean , Modified_csi_std , nni_50_mean , nni_50_std , nni_20_mean , nni_20_std, lf_hf_ratio_mean , lf_hf_ratio_mean , total_power_mean , total_power_std , vlf_mean , vlf_std) , axis = 1)
        



neual_feature,ppg_signal = get_rrfeature(individual = individual , index = neual_index , emotion_type= 'neual' , encode = 0)
positive_feature,ppg_signal = get_rrfeature(individual = individual,index = positive_index , emotion_type= 'positive' , encode = 1)
negetive_feature,ppg_signal = get_rrfeature(individual = individual,index = negetive_index , emotion_type= 'negetive' , encode = 2)

all_feature = pd.concat([neual_feature , positive_feature,negetive_feature],ignore_index=1)
all_feature=all_feature.replace([np.inf, -np.inf], np.nan).dropna(axis=0)#清除nan值
all_feature.to_csv('train_data.csv',index=False)


test_neual_feature,ppg_signal = get_rrfeature(individual = individual,index = test_neual_index , emotion_type= 'neual' , encode = 0)
test_positive_feature,ppg_signal = get_rrfeature(individual = individual,index = test_positive_index , emotion_type= 'positive' , encode = 1)
test_negetive_feature,ppg_signal = get_rrfeature(individual = individual, index = test_negetive_index , emotion_type= 'negetive' , encode = 2)

test_feature = pd.concat([test_neual_feature , test_positive_feature,test_negetive_feature],ignore_index=1)
# test_feature,ppg_signal = get_rrfeature(individual = individual,index = test_neual_index , emotion_type= 'test' , encode = 0)
test_feature=test_feature.replace([np.inf, -np.inf], np.nan).dropna(axis=0)#清除nan值


test_feature.to_csv('test_data.csv',index=False)

df1 = all_feature[all_feature['encode']==0]
df2 = all_feature[all_feature['encode']==1]
df3 = all_feature[all_feature['encode']==2]
# plt.scatter( "max_hr", "min_hr",data=df1, alpha = 0.2)
# plt.scatter("max_hr", "min_hr",data=df2, alpha = 0.2)
# plt.scatter("max_hr", "min_hr", data=df3, alpha = 0.2)
# plt.annotate('neural', xy=(73, 200), xytext=(73, 225), arrowprops = {'color':'green'})
# plt.annotate('positive', xy=(85, 275), xytext=(85, 300), arrowprops = {'color':'blue'})
# plt.annotate('negetive', xy=(80, 255), xytext=(80, 280), arrowprops = {'color':'orange'})
# plt.scatter("range_nni", "mean_hr", data=df1, alpha = 0.2)
# plt.scatter("range_nni", "mean_hr", data=df2, alpha = 0.2)
# plt.scatter("range_nni", "mean_hr", data=df3, alpha = 0.2)
# plt.annotate('neural', xy=(73, 200), xytext=(73, 225), arrowprops = {'color':'green'})
# plt.annotate('positive', xy=(85, 275), xytext=(85, 300), arrowprops = {'color':'blue'})
# plt.annotate('negetive', xy=(80, 255), xytext=(80, 280), arrowprops = {'color':'orange'})
plt.scatter("mean_nni", "median_nni", data=df1, alpha = 0.2)
plt.scatter("mean_nni", "median_nni", data=df2, alpha = 0.2)
plt.scatter("mean_nni", "median_nni", data=df3, alpha = 0.2)
plt.annotate('neural', xy=(73, 200), xytext=(73, 225), arrowprops = {'color':'green'})
plt.annotate('positive', xy=(85, 275), xytext=(85, 300), arrowprops = {'color':'blue'})
plt.annotate('negetive', xy=(80, 255), xytext=(80, 280), arrowprops = {'color':'orange'})
plt.title('feature distribution')
plt.xlabel('max_hr')
plt.ylabel('min_hr')
plt.show()

