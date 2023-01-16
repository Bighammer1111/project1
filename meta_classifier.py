import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from StackingRegressor import StackingRegressor
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.linear_model import LogisticRegression

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

knn = KNeighborsRegressor(n_neighbors=5)
base_learners.append(knn)

rf = forest=RandomForestClassifier(max_depth=5,n_estimators=100,random_state=0)
base_learners.append(rf)

svm1 = svm.SVC(kernel='rbf')

meta_learner = LogisticRegression()

sc = StackingRegressor([[knn,rf,svm1] , [meta_learner]])


sc.fit(all_feature_x , all_feature_y)
meta_data = sc.predict(test_feature_x)
print(meta_data)

base_error = []
base_r2 = []
for i in range(len(base_learners)):
    learner = base_learners[i]
    predictions = meta_data[1][:,i]
    err = metrics.mean_squared_error(test_feature_y,predictions)
    r2 = metrics.accuracy_score(test_feature_y , predictions)
    base_error.append(err)
    base_r2.append(r2)

err = metrics.mean_squared_error(test_feature_y , meta_data[-1])
r2 = metrics.accuracy_score(test_feature_y , meta_data[-1])

print('各個方法效能評估：')
print('-'*20)
for i in range (len(base_learners)):
    e = base_error[i]
    r = base_r2[i]
    b = base_learners[i]
    print(f'{e:.1f} {r:.2f} {b.__class__.__name__}')
print(f'{err:.1f} {r2:.2f} Ensemble')

# kf = KFold(n_splits=5, shuffle=True)

# for i, (train_index, test_index) in enumerate(kf.split(test_feature)):

#     all_feature_x1 = test_feature_x[train_index]
#     all_feature_y1 = test_feature_y[train_index]
#     test_feature_x1 = test_feature_x[test_index]
#     test_feature_y1 = test_feature_y[test_index]
#     sc.fit(all_feature_x1 , all_feature_y1)
#     meta_data = sc.predict(test_feature_x1)

#     base_error = []
#     base_r2 = []
#     for i in range(len(base_learners)):
#         learner = base_learners[i]
#         predictions = meta_data[1][:,i]
#         err = metrics.mean_squared_error(test_feature_y1,predictions)
#         r2 = metrics.accuracy_score(test_feature_y1 , predictions)
#         base_error.append(err)
#         base_r2.append(r2)
#     print(meta_data[-1])
#     err = metrics.mean_squared_error(test_feature_y1 , meta_data[-1])
#     r2 = metrics.accuracy_score(test_feature_y1 , meta_data[-1])

#     print('各個方法效能評估：')
#     print('-'*20)
#     for i in range (len(base_learners)):
#         e = base_error[i]
#         r = base_r2[i]
#         b = base_learners[i]
#         print(f'{e:.1f} {r:.2f} {b.__class__.__name__}')
#     print(f'{err:.1f} {r2:.2f} Ensemble')