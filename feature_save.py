import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from hrvanalysis import get_time_domain_features,remove_outliers

from get_feature_copy import get_rrfeature

all_index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
test_neual_index=[10]
test_positive_index = [10]
test_negetive_index = [10]
neual_index=[1,2,3,4,5,6,7,8,9,10,11,12,13]
positive_index = [1,2,3,4,5,6,7,8,9,10,11,12,13]
negetive_index = [1,2,3,4,5,6,7,8,9,10,11,12,13]
neual_index = np.delete(neual_index,test_neual_index)
positive_index = np.delete(positive_index,test_neual_index)
negetive_index = np.delete(negetive_index,test_neual_index)#將測試資料從訓練資料中刪除
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
sdnn_mean = np.zeros((all_index[-1] , 1))
sdnn_std = np.zeros((all_index[-1] , 1))
sdsd_mean = np.zeros((all_index[-1] , 1))
sdsd_std = np.zeros((all_index[-1] , 1))
rmssd_mean = np.zeros((all_index[-1] , 1))
rmssd_std = np.zeros((all_index[-1] , 1))
nni_50_mean = np.zeros((all_index[-1] , 1))
nni_50_std = np.zeros((all_index[-1] , 1))
nni_20_mean = np.zeros((all_index[-1] , 1))
nni_20_std = np.zeros((all_index[-1] , 1))
cvsd_mean = np.zeros((all_index[-1] , 1))
cvsd_std = np.zeros((all_index[-1] , 1))
cvnni_mean = np.zeros((all_index[-1] , 1))
cvnni_std = np.zeros((all_index[-1] , 1))
std_hr_mean = np.zeros((all_index[-1] , 1))
std_hr_std = np.zeros((all_index[-1] , 1))

for i in all_index:
        file_path = 'F:\\project\\data\\no'+ str(i) + ' baseline.csv'
        data = pd.read_csv(file_path,skiprows=range(1,1000))#刪除資料前段
        rr_interval = data[' RR'].to_numpy()#rr轉numpy
        rr_interval = rr_interval[0:len(rr_interval)].astype(float)#numpy轉array
        

        zero_index = np.where(rr_interval != 0)
        zero_index = np.array(zero_index)[0]
            
        rr_interval= rr_interval[rr_interval != 0]
        rr_interval= remove_outliers(rr_intervals=rr_interval, low_rri=500, high_rri=1500)#去除極端值

        loop= round(len(rr_interval)/8)#以8為單位來切割資料
        index = list(range(1,loop-8))#建立index
        feature = pd.DataFrame(columns=['mean_nni','median_nni','range_nni','mean_hr','max_hr','min_hr','sdnn','sdsd','rmssd','nni_50','nni_20','cvsd','cvnni','std_hr','encode'],index=index)#創建空dataframe
        for j in range(1,loop):
        
        
        # if spo2[i] == 0:
        #     break
        # feature.spo2[i] = spo2[i]

            loop_data = rr_interval[j:j+8]
            time_domain_features = get_time_domain_features(loop_data)#從rr值取出特徵
            feature.mean_nni[j] = time_domain_features['mean_nni'] #從得出的feature存入dataframe
            feature.median_nni[j] = time_domain_features['median_nni'] 
            feature.range_nni[j] = time_domain_features['range_nni'] 
            feature.mean_hr[j] = time_domain_features['mean_hr'] 
            feature.max_hr[j] = time_domain_features['max_hr'] 
            feature.min_hr[j] = time_domain_features['min_hr']  
            feature.sdnn[j] = time_domain_features['sdnn'] 
            feature.sdsd[j] = time_domain_features['sdsd'] 
            feature.rmssd[j] = time_domain_features['rmssd'] 
            feature.nni_50[j] = time_domain_features['nni_50']
            feature.nni_20[j] = time_domain_features['nni_20']
            feature.cvsd[j] = time_domain_features['cvsd'] 
            feature.cvnni[j] = time_domain_features['cvnni']  
            feature.std_hr[j] = time_domain_features['std_hr'] 
        
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
        sdnn_mean[i-1 , 0] =  feature.sdnn.mean()
        sdnn_std[i-1 , 0] =  feature.sdnn.std()
        sdsd_mean[i-1 , 0] =  feature.sdsd.mean()
        sdsd_std[i-1 , 0] =  feature.sdsd.std()
        rmssd_mean[i-1 , 0] =  feature.rmssd.mean()
        rmssd_std[i-1 , 0] =  feature.rmssd.std()
        nni_50_mean[i-1 , 0] =  feature.nni_50.mean()
        nni_50_std[i-1 , 0] =  feature.nni_50.std()
        nni_20_mean[i-1 , 0] =  feature.nni_20.mean()
        nni_20_std[i-1 , 0] =  feature.nni_20.std()
        cvsd_mean[i-1 , 0] =  feature.cvsd.mean()
        cvsd_std[i-1 , 0] =  feature.cvsd.std()
        cvnni_mean[i-1 , 0] =  feature.cvnni.mean()
        cvnni_std[i-1 , 0] =  feature.cvnni.std()
        std_hr_mean[i-1 , 0] =  feature.std_hr.mean()
        std_hr_std[i-1 , 0] =  feature.std_hr.std()

        individual = np.concatenate((mean_nni_mean , mean_nni_std , median_nni_mean , median_nni_std , range_nni_mean , range_nni_std , mean_hr_mean , mean_hr_std , max_hr_mean , max_hr_std , min_hr_mean , min_hr_std
        ,sdnn_mean , sdnn_std , sdsd_mean , sdsd_std , rmssd_mean , rmssd_std , nni_50_mean , nni_50_std , nni_20_mean , nni_20_std , cvsd_mean ,cvsd_std , cvnni_mean , cvnni_std , std_hr_mean , std_hr_std ) , axis = 1)
        



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

