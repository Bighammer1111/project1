
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np
from copy import deepcopy

class StackingRegressor():
    def __init__(self,learners):
        self.level_sizes = []
        self.learners=[]
        for learning_level in learners:
            self.level_sizes.append(len(learning_level))
            level_learners =[]
            for learner in learning_level:
                level_learners.append(deepcopy(learner))
            self.learners.append(level_learners)
    def fit(self,x,y):
        meta_data=[x]
        meta_targets=[y]
        for i in range(len(self.learners)):
            level_size=self.level_sizes[i]

            data_z=np.zeros((level_size,len(x)))
            target_z=np.zeros(len(x))

            traindata_x=meta_data[i]
            traindata_y=meta_targets[i]
           

            KF=KFold(n_splits=5)
            m=0
            for train_indices,test_indices in KF.split(x):
                for j in range(len(self.learners[i])):
                    learner=self.learners[i][j]
                    learner.fit(traindata_x[train_indices],
                                traindata_y[train_indices])

                    p=learner.predict(traindata_x[test_indices])
                    data_z[j][m:m+len(test_indices)]=p
                
                target_z[m:m+
                             len(test_indices)]=traindata_y[test_indices]
                m += len(test_indices)

            data_z=data_z.transpose()
            meta_data.append(data_z)
            meta_targets.append(target_z)

            for learner in self.learners[i]:
                learner.fit(traindata_x,traindata_y)

    def predict(self,x):

        meta_data = [x]
        for i in range(len(self.learners)):
            level_size=self.level_sizes[i]

            data_z=np.zeros((level_size,len(x)))

            test_x=meta_data[i]

            for j in range(len(self.learners[i])):

                learner = self.learners[i][j]
                predictions = learner.predict(test_x)
                data_z[j] = predictions

            data_z=data_z.transpose()
            meta_data.append(data_z)

        return meta_data
