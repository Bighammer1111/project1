import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fftpack import fft
from vmdpy import VMD
from sklearn.ensemble import RandomForestClassifier

from get_feature_select import get_rrfeature,get_1rrfeature

test_neual_index=[7]
test_positive_index = [7]
test_negetive_index = [7]
neual_index=[1,2,3,4,5,6,7,8,10,11,12]
positive_index = [1,2,3,4,5,6,7,8,10,11,12]
negetive_index = [1,2,3,4,5,6,7,8,10,11,12]
neual_index = np.delete(neual_index,test_neual_index)
positive_index = np.delete(positive_index,test_neual_index)
negetive_index = np.delete(negetive_index,test_neual_index)#將測試資料從訓練資料中刪除


neual_feature,ppg_signal = get_rrfeature(index = neual_index , emotion_type= 'neual' , encode = 0)
positive_feature,ppg_signal = get_rrfeature(index = positive_index , emotion_type= 'positive' , encode = 1)
negetive_feature,ppg_signal = get_rrfeature(index = negetive_index , emotion_type= 'negetive' , encode = 2)

all_feature = pd.concat([neual_feature , positive_feature,negetive_feature],ignore_index=1)
print(all_feature)
all_feature=all_feature.dropna(axis = 1)#清除nan值

test_neual_feature,ppg_signal = get_1rrfeature(index = test_neual_index , emotion_type= 'neual' , encode = 0)
test_positive_feature,ppg_signal = get_1rrfeature(index = test_positive_index , emotion_type= 'positive' , encode = 1)
test_negetive_feature,ppg_signal = get_1rrfeature(index = test_negetive_index , emotion_type= 'negetive' , encode = 2)

test_feature = pd.concat([test_neual_feature , test_positive_feature,test_negetive_feature],ignore_index=1)
print(all_feature)
test_feature=test_feature.dropna(axis = 1)#清除nan值

# print(neual_feature)
# plt.plot(ppg_signal)
# plt.show()

all_feature_x=all_feature.drop(['encode'],axis=1)#將encode去掉，方便訓練
all_feature_y=all_feature['encode']
all_feature_y=all_feature_y.astype('int')

test_feature_x=test_feature.drop(['encode'],axis=1)#將encode去掉，方便訓練
test_feature_y=test_feature['encode']
test_feature_y=test_feature_y.astype('int')



forest=RandomForestClassifier(max_depth=18,n_estimators=100,random_state=0)
forest.fit(all_feature_x,all_feature_y)
prediction=forest.predict(test_feature_x)

print(forest.score(test_feature_x,test_feature_y))