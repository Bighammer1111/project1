import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
# from scipy.fftpack import fft
# from vmdpy import VMD
# from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


all_feature = pd.read_csv('train_data.csv')
test_feature = pd.read_csv('test_data.csv')


all_feature_x=all_feature.drop(['encode'],axis=1)#將encode去掉，方便訓練
all_feature_x = all_feature_x.drop(['mean_nni','median_nni','range_nni','mean_hr','max_hr','min_hr','nni_50','nni_20','csi','cvi'
                                        ,'Modified_csi','lf_hf_ratio','total_power','vlf'] , axis=1)
all_feature_y=all_feature['encode']
all_feature_y=all_feature_y.astype('int')
all_feature_x = all_feature_x.values
all_feature_y = all_feature_y.values

test_feature_x=test_feature.drop(['encode'],axis=1)#將encode去掉，方便訓練
# test_feature_x = test_feature_x.drop(['freq_mean' , 'freq_std'] , axis=1)
test_feature_x = test_feature_x.drop(['mean_nni','median_nni','range_nni','mean_hr','max_hr','min_hr','nni_50','nni_20','csi','cvi'
                                        ,'Modified_csi','lf_hf_ratio','total_power','vlf'] , axis=1)

test_feature_y=test_feature['encode']
test_feature_y=test_feature_y.astype('int')
test_feature_x = test_feature_x.values
test_feature_y = test_feature_y.values

# kf = KFold(n_splits=3 , shuffle=True)

# for i, (train_index, test_index) in enumerate(kf.split(all_feature_x)):

#     kfold_train_x = all_feature_x[train_index]
#     kfold_train_y = all_feature_y[train_index]
#     kfold_test_x = all_feature_x[test_index]
#     kfold_test_y = all_feature_y[test_index]
#     ensemble_size = 500
#     ensemble = XGBClassifier(n_estimators = ensemble_size , n_jobs = 10)
#     ensemble.fit(kfold_train_x , kfold_train_y)

#     ensemble_predictions = ensemble.predict(kfold_test_x)
#     ensemble_predictions2 = ensemble.predict(kfold_train_x)
#     ensemble_acc = metrics.accuracy_score(kfold_test_y , ensemble_predictions)
#     ensemble_acc2 = metrics.accuracy_score(kfold_train_y , ensemble_predictions2)

#     print('Test_data: %.2f' % ensemble_acc)
#     print('Train_data: %.2f' % ensemble_acc2)

#     cm = confusion_matrix(kfold_test_y,ensemble_predictions , labels=[0,1,2])
#     fig, px = plt.subplots(figsize=(7.5, 7.5))
#     px.matshow(cm, cmap=plt.cm.YlOrRd, alpha=0.5)
#     for m in range(cm.shape[0]):
#         for n in range(cm.shape[1]):
#             px.text(x=m,y=n,s=cm[m, n], va='center', ha='center', size='xx-large')

# # Sets the labels
#     plt.xlabel('Predictions', fontsize=16)
#     plt.ylabel('Actuals', fontsize=16)
#     plt.title('Confusion Matrix', fontsize=15)
#     plt.show()

ensemble_size = 100
ensemble = XGBClassifier(n_estimators = ensemble_size , n_jobs = 10 ,max_depth = 20)

kf = KFold(n_splits=3, shuffle=True)



ensemble.fit(all_feature_x , all_feature_y)

ensemble_predictions = ensemble.predict(test_feature_x)
ensemble_predictions2 = ensemble.predict(all_feature_x)
ensemble_acc = metrics.accuracy_score(test_feature_y , ensemble_predictions)
ensemble_acc2 = metrics.accuracy_score(all_feature_y , ensemble_predictions2)

print('Test_data: %.2f' % ensemble_acc)
print('Train_data: %.2f' % ensemble_acc2)

cm = confusion_matrix(ensemble_predictions,test_feature_y , labels=[0,1,2])
fig, px = plt.subplots(figsize=(7.5, 7.5))
px.matshow(cm, cmap=plt.cm.YlOrRd, alpha=0.5)
for m in range(cm.shape[0]):
    for n in range(cm.shape[1]):
        px.text(x=m,y=n,s=cm[m, n], va='center', ha='center', size='xx-large')

# Sets the labels
plt.xlabel('Predictions', fontsize=16)
plt.ylabel('Actuals', fontsize=16)
plt.title('Confusion Matrix', fontsize=15)
plt.show()