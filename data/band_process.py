import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fftpack import fft
from vmdpy import VMD
from hrvanalysis import get_time_domain_features
from hrvanalysis import plot_psd

data = pd.read_csv('no1 negetive.csv',skiprows=range(1,1500))#刪除資料中重複的索引值
for i in range (0,len(data)):
    if data['Time'][i] == 'Time':
        data = data.drop([i])

rr_interval = data[' RR'].to_numpy()#rr轉numpy

rr_interval = rr_interval[0:len(rr_interval)].astype(float)#numpy轉array

zero_index = np.where(rr_interval != 0)
zero_index = np.array(zero_index)[0]
spo2 = data[' SpO2 (%)'][zero_index]
b= rr_interval[rr_interval != 0]

print(b)
print(spo2)