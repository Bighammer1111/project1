import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from StackingRegressor import StackingRegressor
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

all_feature = pd.read_csv('train_data.csv')
test_feature = pd.read_csv('test_data.csv')

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

base_learners = []

knn = KNeighborsRegressor(n_neighbors=1)
base_learners.append(knn)

rf=RandomForestClassifier(max_depth=15,n_estimators=70,random_state=0)
base_learners.append(rf)

MLP1 = MLPClassifier(hidden_layer_sizes=(100,) , random_state=2)
base_learners.append(MLP1)
# svm1 = Ridge()

meta_learner = RandomForestClassifier(n_estimators=100,random_state=0)

sc = StackingRegressor([[rf,knn,MLP1] , [meta_learner]])


# sc.fit(all_feature_x , all_feature_y) 
# meta_data = sc.predict(test_feature_x)


# base_error = []
# base_r2 = []
# for i in range(len(base_learners)):
#     learner = base_learners[i]
#     predictions = meta_data[1][:,i]
#     err = metrics.mean_squared_error(test_feature_y,predictions)
#     r2 = metrics.r2_score(test_feature_y , predictions)
#     base_error.append(err)
#     base_r2.append(r2)

# err = metrics.mean_squared_error(test_feature_y , meta_data[-1])
# r2 = metrics.accuracy_score(test_feature_y , meta_data[-1])

# print('各個方法效能評估：')
# print('-'*20)
# for i in range (len(base_learners)):
#     e = base_error[i]
#     r = base_r2[i]
#     b = base_learners[i]
#     print(f'{e:.1f} {r:.2f} {b.__class__.__name__}')
# print(f'{err:.1f} {r2:.2f} Ensemble')

# pred = np.around(meta_data[-1])

# cm = confusion_matrix(test_feature_y, pred, labels=[0,1,2])
# fig, px = plt.subplots(figsize=(7.5, 7.5))
# px.matshow(cm, cmap=plt.cm.YlOrRd, alpha=0.5)
# for m in range(cm.shape[0]):
#     for n in range(cm.shape[1]):
#         px.text(x=m,y=n,s=cm[m, n], va='center', ha='center', size='xx-large')
# plt.xlabel('Predictions', fontsize=16)
# plt.ylabel('Actuals', fontsize=16)
# plt.title('Confusion Matrix', fontsize=15)
# plt.show()

kf = KFold(n_splits=5, shuffle=True)

for i, (train_index, test_index) in enumerate(kf.split(all_feature_x)):

    all_feature_x1 = all_feature_x[train_index]
    all_feature_y1 = all_feature_y[train_index]
    test_feature_x1 = all_feature_x[test_index]
    test_feature_y1 = all_feature_y[test_index]
    sc.fit(all_feature_x1 , all_feature_y1)
    meta_data = sc.predict(test_feature_x1)

    # 訓練、預測
    sc.fit(all_feature_x1, all_feature_y1)
    meta_data = sc.predict(test_feature_x1)
    print(meta_data)

    # 衡量基學習器跟集成後效能
    base_errors = []
    base_r2 = []
    for i in range(len(base_learners)):
        learner = base_learners[i]
        predictions = meta_data[1][:,i]
        err = metrics.mean_squared_error(test_feature_y1, predictions)
        r2 = metrics.accuracy_score(test_feature_y1, predictions)
        base_errors.append(err)
        base_r2.append(r2)

    err = metrics.mean_squared_error(test_feature_y1, meta_data[-1])
    r2 = metrics.accuracy_score(test_feature_y1, meta_data[-1])

    # 顯示結果
    print('ERROR  R2  Name')
    print('-'*20)
    for i in range(len(base_learners)):
        e = base_errors[i]
        r = base_r2[i]
        b = base_learners[i]
        print(f'{e:.1f} {r:.2f} {b.__class__.__name__}')
    print(f'{err:.1f} {r2:.2f} Ensemble')