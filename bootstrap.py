from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits 
from sklearn import metrics
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


np.random.seed(1)
all_feature = pd.read_csv('train_data.csv')
test_feature = pd.read_csv('test_data.csv')

all_feature_x=all_feature.drop(['encode'],axis=1)#將encodeeeeee去掉，方便訓練
# all_feature_x = all_feature_x.drop(['freq_mean' , 'freq_std'] , axis=1)
all_feature_y=all_feature['encode']
all_feature_y=all_feature_y.astype('int')
all_feature_x = all_feature_x.values
all_feature_y = all_feature_y.values

test_feature_x=test_feature.drop(['encode'],axis=1)#將encode去掉，方便訓練
# test_feature_x = test_feature_x.drop(['freq_mean' , 'freq_std'] , axis=1)
test_feature_y=test_feature['encode']
test_feature_y=test_feature_y.astype('int')
test_x = test_feature_x.values
test_y = test_feature_y.values



kf = KFold(n_splits=3 , shuffle=True)

for i, (train_index, test_index) in enumerate(kf.split(test_feature_x)):
    train_size = 8529
    # train_x = all_feature_x[train_index]
    # train_y = all_feature_y[train_index]
    # test_x = all_feature_x[test_index]
    # test_y = all_feature_y[test_index]
    train_x = all_feature_x
    train_y = all_feature_y
    test_x = test_x
    test_y = test_y


    


    # --- 第 2 部分 ---
    # 產生子樣本並訓練基學習器
    ensemble_size = 30
    base_learners = []

    for _ in range(ensemble_size):
        # 產生子樣本
        bootstrap_sample_indices = np.random.randint(0, train_size, size = train_size)
        bootstrap_x = train_x[bootstrap_sample_indices]
        bootstrap_y = train_y[bootstrap_sample_indices]
        # 訓練基學習器
        dtree = DecisionTreeClassifier()
        dtree.fit(bootstrap_x, bootstrap_y)
        base_learners.append(dtree)

    # --- 第 3 部分 ---
    # 用基學習器做預測並評估效能
    base_predictions = []
    base_accuracy = []
    for learner in base_learners:
        predictions = learner.predict(test_x)
        base_predictions.append(predictions)
        acc = metrics.accuracy_score(test_y, predictions)
        base_accuracy.append(acc)
    # --- 第 4 部分 ---
    # 組合基學習器的預測

    ensemble_predictions = []
    # 找出每一筆資料得票最多的類別
    for i in range(len(test_y)):
        # 計算每個類別的得票數
        counts = [0 for _ in range(10)]
        for learner_p in base_predictions:
            counts[learner_p[i]] = counts[learner_p[i]] + 1
        # 找到得票最多的類別
        final = np.argmax(counts)
        # 將此類別加入最終預測中
        ensemble_predictions.append(final)

    ensemble_acc = metrics.accuracy_score(test_y, 
                                        ensemble_predictions)

    # --- 第 5 部分 ---
    # 顯示準確率
    print('Base Learners:')
    print('-'*30)
    for index, acc in enumerate(sorted(base_accuracy)):
        print(f'Learner {index+1}: %.2f' % acc)
    print('-'*30)
    print('Bagging: %.2f' % ensemble_acc)
    # --- 第 5 部分 ---
    # 顯示準確率
    print('Base Learners:')
    print('-'*30)
    for index, acc in enumerate((base_accuracy)):
        print(f'Learner {index+1}: %.2f' % acc)
    print('-'*30)
    print('Bagging: %.2f' % ensemble_acc)

    pred = ensemble_predictions

    cm = confusion_matrix(ensemble_predictions,test_feature_y , labels=[0,1,2])
    fig, px = plt.subplots(figsize=(7.5, 7.5))
    px.matshow(cm, cmap=plt.cm.YlOrRd, alpha=0.5)
    for m in range(cm.shape[0]):
        for n in range(cm.shape[1]):
            px.text(x=m,y=n,s=cm[m, n], va='center', ha='center', size='xx-large')
    plt.xlabel('Predictions', fontsize=16)
    plt.ylabel('Actuals', fontsize=16)
    plt.title('Confusion Matrix', fontsize=15)
    plt.show()