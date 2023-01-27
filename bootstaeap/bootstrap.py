# --- 第 1 部分 ---
# 載入函式庫與資料集
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np
digits = load_digits()

#np.random.seed(1)
train_size = 1500
train_x, train_y = digits.data[:train_size], digits.target[:train_size]
test_x, test_y = digits.data[train_size:], digits.target[train_size:]

 # --- 第 2 部分 ---
# 產生子樣本並訓練基學習器
ensemble_size = 10
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