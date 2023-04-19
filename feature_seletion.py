#本程式為測試各個特徵的有效程度

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('F:\\project\\train_data.csv',index_col=0)
y = data['encode']
X = data.drop(['encode'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=100,test_size=0.3)   

cor = X_train.corr() #相關矩陣
plt.figure(figsize=(12,10))
sns.heatmap(cor, cmap=plt.cm.CMRmap_r,annot=True)
plt.show()  

from sklearn.feature_selection import mutual_info_classif#特徵選擇過濾器
# determine the mutual information
mutual_info = mutual_info_classif(X_train, y_train)
mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
mutual_info.sort_values(ascending=False)
mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 8))
plt.show()

from sklearn.feature_selection import SelectKBest#選出k個最佳feature
#No we Will select the top 5 important features
sel_five_cols = SelectKBest(mutual_info_classif, k=10)
sel_five_cols.fit(X_train, y_train)
print(X_train.columns[sel_five_cols.get_support()])