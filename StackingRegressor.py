
# --- 第 1 部分 ---
# 匯入函式庫
import numpy as np
from sklearn.model_selection import KFold
from copy import deepcopy
# --- 第 2 部分 ---
class StackingRegressor():
    def __init__(self, learners):
        # 接收基學習器、超學習器、以及堆疊中每一層分別有多少學習器
        # 複製學習器
        self.level_sizes = []
        self.learners = []
        for learning_level in learners:
            self.level_sizes.append(len(learning_level))
            level_learners = []
            for learner in learning_level:
                level_learners.append(deepcopy(learner))
            self.learners.append(level_learners)
# --- 第 3 部分 ---
    # fit 函式
    # 用第i-1層的基學習器預測值來訓練第i層的基學習器
    def fit(self, x, y):
        # 第1層基學習器的訓練資料即為原始資料
        meta_data = [x]
        meta_targets = [y]
        for i in range(len(self.learners)):
            level_size = self.level_sizes[i]

            # 建立第i層預測值的儲存空間
            data_z = np.zeros((level_size, len(x)))
            target_z = np.zeros(len(x))

            # 取得第i層訓練資料集
            train_x = meta_data[i]
            train_y = meta_targets[i]

            # 建立交叉驗證
            KF = KFold(n_splits=5)
            m = 0
            for train_indices, test_indices in KF.split(x):
                for j in range(len(self.learners[i])):
                    # 使用前K-1折訓練第j個基學習器
                    learner = self.learners[i][j]
                    learner.fit(train_x[train_indices], 
                                train_y[train_indices])
                    # 使用第K折驗證第j個基學習器
                    p = learner.predict(train_x[test_indices])
                    # 儲存第K折第j個基學習器預測結果
                    data_z[j][m:m+len(test_indices)] = p

                target_z[m:m + 
                    len(test_indices)] = train_y[test_indices]
                m += len(test_indices)

            # 儲存第i層基學習器的預測結果
            # 作為第i+1層基學習器的訓練資料
            data_z = data_z.transpose()
            meta_data.append(data_z)
            meta_targets.append(target_z)


            # 使用完整的訓練資料來訓練基學習器
            for learner in self.learners[i]:
                learner.fit(train_x, train_y)
# --- 第 4 部分 ---
    # predict 函式
    def predict(self, x):

        # 儲存每一層的預測
        meta_data = [x]
        for i in range(len(self.learners)):
            level_size = self.level_sizes[i]

            data_z = np.zeros((level_size, len(x)))

            test_x = meta_data[i]

            for j in range(len(self.learners[i])):

                learner = self.learners[i][j]
                predictions = learner.predict(test_x)
                data_z[j] = predictions

            # 儲存第i層基學習器的預測結果
            # 作為第i+1層基學習器的輸入
            data_z = data_z.transpose()
            meta_data.append(data_z)

        # 傳回預測結果
        return meta_data