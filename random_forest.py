import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fftpack import fft
from vmdpy import VMD
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import metrics

all_feature = pd.read_csv('train_data.csv')
test_feature = pd.read_csv('test_data.csv')


# print(neual_feature)
# plt.plot(ppg_signal)
# plt.show()


all_feature_x=all_feature.drop(['encode'],axis=1)#將encode去掉，方便訓練
all_feature_y=all_feature['encode']
all_feature_y=all_feature_y.astype('int')
all_feature_x = all_feature_x.values
all_feature_y = all_feature_y.values

test_feature_x=test_feature.drop(['encode'],axis=1)#將encode去掉，方便訓練
test_feature_y=test_feature['encode']
test_feature_y=test_feature_y.astype('int')
test_feature_x = test_feature_x.values
test_feature_y = test_feature_y.values



kfold_train_x = all_feature_x
kfold_train_y = all_feature_y
kfold_test_x = test_feature_x
kfold_test_y = test_feature_y
# kfold_train_x = all_feature_x[train_index]
# kfold_train_y = all_feature_y[train_index]
# kfold_test_x = all_feature_x[test_index]
# kfold_test_y = all_feature_y[test_index]


# forest=RandomForestClassifier(max_depth=20,n_estimators=70,random_state=0)
# forest= SVC(kernel='poly', C = 10.0)
forest=KNeighborsRegressor(n_neighbors=3)

# forest.fit(all_feature_x,all_feature_y)
# prediction=forest.predict(test_feature_x)

# print(forest.score(test_feature_x,test_feature_y))
forest.fit(kfold_train_x,kfold_train_y)
prediction=forest.predict(kfold_test_x)

print(prediction)
prediction = np.round(prediction)

cm = confusion_matrix(prediction,test_feature_y , labels=[0,1,2])
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
print(metrics.accuracy_score(test_feature_y , prediction))
print(metrics.precision_score(test_feature_y , prediction,average="macro"))
print(metrics.recall_score(test_feature_y , prediction,average="macro"))
print(metrics.f1_score(test_feature_y , prediction,average="macro"))

